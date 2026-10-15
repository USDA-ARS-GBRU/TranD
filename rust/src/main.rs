use anyhow::{Context, Result};
use clap::Parser;
use polars::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

/// CLI
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "TranD (two-GTF) pairwise transcript distance metrics in Rust — junction/ER/EF/nt + AS flags"
)]
struct Args {
    /// Path to first GTF (exon records)
    #[arg(long)]
    gtf1: PathBuf,
    /// Path to second GTF (exon records)
    #[arg(long)]
    gtf2: PathBuf,
    /// Output Parquet file for pairwise distances
    #[arg(long, default_value = "transcript_distance.parquet")]
    out: PathBuf,
    /// Include non-overlapping transcript pairs in output (flag_no_ovlp_nt = 1)
    #[arg(long, default_value_t = false)]
    include_zero_overlap: bool,
}

/// Parsed exon record (GTF exon lines)
#[derive(Debug, Clone)]
struct ExonRecord {
    seqname: String,
    strand: String,
    start: i64,
    end: i64,
    gene_id: String,
    transcript_id: String,
}

/// Transcript model (per transcript)
#[derive(Debug, Clone)]
struct TranscriptModel {
    seqname: String,
    strand: String,
    gene_id: String,
    transcript_id: String,
    min_start: i64,
    max_end: i64,
    exons: Vec<(i64, i64)>,   // merged, sorted, disjoint
    junctions: Vec<String>,   // ordered donor/acceptor coordinates
    introns: Vec<(i64, i64)>, // exon-to-exon gaps
}

/// Parse exon lines from a GTF file (only 'exon' feature rows).
/// Extract gene_id/transcript_id/seqname/strand/start/end from attributes.
fn parse_gtf_exons(path: &PathBuf) -> Result<Vec<ExonRecord>> {
    let file = File::open(path).with_context(|| format!("Failed to open GTF: {:?}", path))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 9 || parts[2] != "exon" {
            continue;
        }
        let seqname = parts[0].to_string();
        let start: i64 = parts[3].parse().unwrap_or(0);
        let end: i64 = parts[4].parse().unwrap_or(0);
        let strand = parts[6].to_string();
        let attrs = parts[8];
        let mut gid = None;
        let mut tid = None;
        for attr in attrs.split(';') {
            let a = attr.trim();
            if a.starts_with("gene_id") {
                if let Some(val) = a.split_whitespace().nth(1) {
                    gid = Some(val.trim_matches('"').to_string());
                }
            } else if a.starts_with("transcript_id") {
                if let Some(val) = a.split_whitespace().nth(1) {
                    tid = Some(val.trim_matches('"').to_string());
                }
            }
        }
        if let (Some(gene_id), Some(transcript_id)) = (gid, tid) {
            out.push(ExonRecord {
                seqname,
                strand,
                start,
                end,
                gene_id,
                transcript_id,
            });
        }
    }
    Ok(out)
}

/// Merge overlapping intervals into disjoint, sorted blocks (used for exons).
fn merge_intervals(mut intervals: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    if intervals.is_empty() {
        return Vec::new();
    }
    intervals.sort_by_key(|x| x.0);
    let mut merged = vec![intervals[0]];
    for (s, e) in intervals.into_iter().skip(1) {
        let last = merged.len() - 1;
        if s <= merged[last].1 {
            // Mutate the tail end directly to avoid overlapping borrows
            merged[last].1 = merged[last].1.max(e);
        } else {
            merged.push((s, e));
        }
    }
    merged
}

/// Build per-transcript model: merged exons; ordered junctions; introns.
/// Junction coords: left_end = exon_end - 10; right_start = next_exon_start + 10.
fn build_transcript_model(
    gid: String,
    tid: String,
    raw_exons: Vec<(i64, i64)>,
    seqname: String,
    strand: String,
) -> TranscriptModel {
    let exons = merge_intervals(raw_exons);
    let min_start = exons.iter().map(|x| x.0).min().unwrap_or(0);
    let max_end = exons.iter().map(|x| x.1).max().unwrap_or(0);

    let mut junctions = Vec::new();
    let mut introns = Vec::new();
    for i in 0..exons.len().saturating_sub(1) {
        let left_end = exons[i].1 - 10;
        let right_start = exons[i + 1].0 + 10;
        junctions.push(format!("{}:{}:{}:{}", seqname, left_end, right_start, strand));
        introns.push((exons[i].1, exons[i + 1].0));
    }

    TranscriptModel {
        seqname,
        strand,
        gene_id: gid,
        transcript_id: tid,
        min_start,
        max_end,
        exons,
        junctions,
        introns,
    }
}

/// EF metadata per ER fragment
#[derive(Debug)]
struct EfInfo {
    id: String,
    er_id: String,
    start: i64,
    end: i64,
    in_t1: bool,
    in_t2: bool,
    ir_t1: bool,
    ir_t2: bool,
}

/// Pairwise metrics (TranD’s td_df_cols)
#[allow(clippy::too_many_fields)]
struct PairMetrics {
    // Junctions
    num_jxn_only_t1: usize,
    num_jxn_only_t2: usize,
    num_jxn_ovlp: usize,
    prop_jxn_no_ovlp: f64,
    prop_jxn_ovlp: f64,
    jxn_string_t1: String,
    jxn_string_t2: String,
    jxn_only_t1: String,
    jxn_only_t2: String,
    jxn_same: String,
    // ER
    num_er_only_t1: usize,
    num_er_only_t2: usize,
    num_er_ovlp: usize,
    prop_er_no_ovlp: f64,
    prop_er_ovlp: f64,
    er_only_t1: String,
    er_only_t2: String,
    er_ovlp: String,
    // EF + exon counts
    num_ef_only_t1: usize,
    num_ef_only_t2: usize,
    num_ef_ovlp: usize,
    prop_ef_no_ovlp: f64,
    prop_ef_ovlp: f64,
    ef_only_t1: String,
    ef_only_t2: String,
    ef_ovlp: String,
    num_exon_only_t1: usize,
    num_exon_only_t2: usize,
    num_exon_ovlp: usize,
    // IR
    num_ir_ef_t1: usize,
    num_ir_ef_t2: usize,
    ir_ef_t1: String,
    ir_ef_t2: String,
    // Nucleotides (global + shared-ER restricted)
    num_nt_ovlp: i64,
    num_nt_only_t1: i64,
    num_nt_only_t2: i64,
    num_nt_no_ovlp: i64,
    total_nt: i64,
    prop_nt_no_ovlp: f64,
    prop_nt_ovlp: f64,
    num_nt_only_t1_in_ovlp_er: i64,
    num_nt_only_t2_in_ovlp_er: i64,
    num_nt_ovlp_in_ovlp_er: i64,
    total_nt_in_ovlp_er: i64,
    prop_nt_no_ovlp_in_ovlp_er: f64,
    prop_nt_ovlp_in_ovlp_er: f64,
    num_nt_only_t1_in_uniq_er: i64,
    num_nt_only_t2_in_uniq_er: i64,
    // Flags
    flag_fsm: i32,
    flag_t1_ism_of_t2: i32,
    flag_t2_ism_of_t1: i32,
    flag_ir: i32,
    flag_5_var: i32,
    flag_3_var: i32,
    flag_alt_da: i32,
    flag_alt_exon: i32,
    flag_no_ovlp_nt: i32,
}

/// Compute pairwise metrics (T1 vs T2)
fn compute_pairwise_metrics(t1: &TranscriptModel, t2: &TranscriptModel) -> PairMetrics {
    // -------- 1) Junctions
    let jxn_string_t1 = t1.junctions.join("|");
    let jxn_string_t2 = t2.junctions.join("|");

    let set_j1: HashSet<&String> = t1.junctions.iter().collect();
    let set_j2: HashSet<&String> = t2.junctions.iter().collect();

    let shared_jxns: Vec<&String> = t1.junctions.iter().filter(|j| set_j2.contains(*j)).collect();
    let only_t1_jxns: Vec<&String> = t1.junctions.iter().filter(|j| !set_j2.contains(*j)).collect();
    let only_t2_jxns: Vec<&String> = t2.junctions.iter().filter(|j| !set_j1.contains(*j)).collect();

    let num_jxn_ovlp = shared_jxns.len();
    let num_jxn_only_t1 = only_t1_jxns.len();
    let num_jxn_only_t2 = only_t2_jxns.len();

    let total_jxn_union = num_jxn_ovlp + num_jxn_only_t1 + num_jxn_only_t2;
    let (prop_jxn_no_ovlp, prop_jxn_ovlp) = if total_jxn_union == 0 {
        (0.0, 1.0)
    } else {
        let no = (num_jxn_only_t1 + num_jxn_only_t2) as f64 / total_jxn_union as f64;
        (no, 1.0 - no)
    };

    // Join as &str (no double clone)
    let jxn_only_t1 = only_t1_jxns
        .into_iter()
        .map(|s| s.as_str())
        .collect::<Vec<&str>>()
        .join("|");
    let jxn_only_t2 = only_t2_jxns
        .into_iter()
        .map(|s| s.as_str())
        .collect::<Vec<&str>>()
        .join("|");
    let jxn_same = shared_jxns
        .into_iter()
        .map(|s| s.as_str())
        .collect::<Vec<&str>>()
        .join("|");

    // -------- 2) ER + EF
    // ER: merge union of T1+T2 exons
    let mut all_exons = t1.exons.clone();
    all_exons.extend_from_slice(&t2.exons);
    let ers = merge_intervals(all_exons);

    // EF boundaries: union of every exon start/end across both transcripts
    let mut boundaries: Vec<i64> = Vec::new();
    for &(s, e) in &t1.exons {
        boundaries.push(s);
        boundaries.push(e);
    }
    for &(s, e) in &t2.exons {
        boundaries.push(s);
        boundaries.push(e);
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    // Helpers
    let contains_fragment = |start: i64, end: i64, exons: &[(i64, i64)]| -> bool {
        exons.iter().any(|&(es, ee)| es <= start && end <= ee)
    };
    let matches_intron = |start: i64, end: i64, introns: &[(i64, i64)]| -> bool {
        introns.iter().any(|&(is, ie)| is == start && ie == end)
    };

    let mut ef_list: Vec<EfInfo> = Vec::new();
    let mut er_ef_counts: HashMap<String, usize> = HashMap::new();

    // Track ER overlap presence
    let mut er_t1_has: HashSet<String> = HashSet::new();
    let mut er_t2_has: HashSet<String> = HashSet::new();

    for &(er_s, er_e) in &ers {
        let er_id = format!("{}:{}:{}:{}", t1.seqname, er_s, er_e, t1.strand);

        // Boundaries inside this ER (ensure both ends are included)
        let mut local_boundaries: Vec<i64> = boundaries
            .iter()
            .copied()
            .filter(|&b| b >= er_s && b <= er_e)
            .collect();
        if local_boundaries.is_empty() || *local_boundaries.first().unwrap() != er_s {
            local_boundaries.insert(0, er_s);
        }
        if *local_boundaries.last().unwrap() != er_e {
            local_boundaries.push(er_e);
        }

        // Internal EF test (for IR): EF must not touch ER ends
        let internal = |b0: i64, b1: i64| b0 > er_s && b1 < er_e;

        for w in local_boundaries.windows(2) {
            let (b0, b1) = (w[0], w[1]);
            if b0 >= b1 {
                continue;
            }

            let in1 = contains_fragment(b0, b1, &t1.exons);
            let in2 = contains_fragment(b0, b1, &t2.exons);
            if !in1 && !in2 {
                continue;
            }

            let ef_id = format!("{}:{}:{}:{}", t1.seqname, b0, b1, t1.strand);

            if in1 {
                er_t1_has.insert(er_id.clone());
            }
            if in2 {
                er_t2_has.insert(er_id.clone());
            }

            // IR marking: unique, internal EF that matches intron coords of the other transcript
            let ir_t1 = in1 && !in2 && internal(b0, b1) && matches_intron(b0, b1, &t2.introns);
            let ir_t2 = in2 && !in1 && internal(b0, b1) && matches_intron(b0, b1, &t1.introns);

            ef_list.push(EfInfo {
                id: ef_id.clone(),
                er_id: er_id.clone(),
                start: b0,
                end: b1,
                in_t1: in1,
                in_t2: in2,
                ir_t1,
                ir_t2,
            });
            *er_ef_counts.entry(er_id.clone()).or_default() += 1;
        }
    }

    // ER summaries: unique/shared
    let mut er_only_t1_list = Vec::new();
    let mut er_only_t2_list = Vec::new();
    let mut er_ovlp_list = Vec::new();

    for &(er_s, er_e) in &ers {
        let er_id = format!("{}:{}:{}:{}", t1.seqname, er_s, er_e, t1.strand);
        let has1 = er_t1_has.contains(&er_id);
        let has2 = er_t2_has.contains(&er_id);
        if has1 && has2 {
            er_ovlp_list.push(er_id);
        } else if has1 {
            er_only_t1_list.push(er_id);
        } else if has2 {
            er_only_t2_list.push(er_id);
        }
    }

    let num_er_only_t1 = er_only_t1_list.len();
    let num_er_only_t2 = er_only_t2_list.len();
    let num_er_ovlp = er_ovlp_list.len();
    let total_er_union = num_er_only_t1 + num_er_only_t2 + num_er_ovlp;
    let (prop_er_no_ovlp, prop_er_ovlp) = if total_er_union == 0 {
        (0.0, 1.0)
    } else {
        let no = (num_er_only_t1 + num_er_only_t2) as f64 / total_er_union as f64;
        (no, 1.0 - no)
    };

    // EF + nucleotide summaries
    let shared_er: HashSet<&String> = er_ovlp_list.iter().collect();
    let mut ef_only_t1_list = Vec::new();
    let mut ef_only_t2_list = Vec::new();
    let mut ef_ovlp_list = Vec::new();

    let mut num_exon_only_t1 = 0;
    let mut num_exon_only_t2 = 0;
    let mut num_exon_ovlp = 0;

    let mut ir_ef_t1_list = Vec::new();
    let mut ir_ef_t2_list = Vec::new();

    let mut num_nt_ovlp: i64 = 0;
    let mut num_nt_only_t1: i64 = 0;
    let mut num_nt_only_t2: i64 = 0;
    let mut num_nt_only_t1_in_ovlp_er: i64 = 0;
    let mut num_nt_only_t2_in_ovlp_er: i64 = 0;
    let mut num_nt_ovlp_in_ovlp_er: i64 = 0;
    let mut num_nt_only_t1_in_uniq_er: i64 = 0;
    let mut num_nt_only_t2_in_uniq_er: i64 = 0;

    for ef in &ef_list {
        let len = ef.end - ef.start;
        let is_singleton = er_ef_counts.get(&ef.er_id).copied().unwrap_or(0) == 1;
        let in_shared_er = shared_er.contains(&ef.er_id);

        if ef.ir_t1 {
            ir_ef_t1_list.push(ef.id.clone());
        }
        if ef.ir_t2 {
            ir_ef_t2_list.push(ef.id.clone());
        }

        match (ef.in_t1, ef.in_t2) {
            (true, true) => {
                ef_ovlp_list.push(ef.id.clone());
                if is_singleton {
                    num_exon_ovlp += 1;
                }
                num_nt_ovlp += len;
                if in_shared_er {
                    num_nt_ovlp_in_ovlp_er += len;
                }
            }
            (true, false) => {
                ef_only_t1_list.push(ef.id.clone());
                if is_singleton {
                    num_exon_only_t1 += 1;
                }
                num_nt_only_t1 += len;
                if in_shared_er {
                    num_nt_only_t1_in_ovlp_er += len;
                } else {
                    num_nt_only_t1_in_uniq_er += len;
                }
            }
            (false, true) => {
                ef_only_t2_list.push(ef.id.clone());
                if is_singleton {
                    num_exon_only_t2 += 1;
                }
                num_nt_only_t2 += len;
                if in_shared_er {
                    num_nt_only_t2_in_ovlp_er += len;
                } else {
                    num_nt_only_t2_in_uniq_er += len;
                }
            }
            _ => {}
        }
    }

    let num_ef_only_t1 = ef_only_t1_list.len();
    let num_ef_only_t2 = ef_only_t2_list.len();
    let num_ef_ovlp = ef_ovlp_list.len();
    let total_ef_union = num_ef_only_t1 + num_ef_only_t2 + num_ef_ovlp;
    let (prop_ef_no_ovlp, prop_ef_ovlp) = if total_ef_union == 0 {
        (0.0, 1.0)
    } else {
        let no = (num_ef_only_t1 + num_ef_only_t2) as f64 / total_ef_union as f64;
        (no, 1.0 - no)
    };

    let num_nt_no_ovlp = num_nt_only_t1 + num_nt_only_t2;
    let total_nt = num_nt_ovlp + num_nt_no_ovlp;
    let (prop_nt_no_ovlp, prop_nt_ovlp) = if total_nt == 0 {
        (0.0, 1.0)
    } else {
        let no = num_nt_no_ovlp as f64 / total_nt as f64;
        (no, 1.0 - no)
    };

    let total_nt_in_ovlp_er =
        num_nt_ovlp_in_ovlp_er + num_nt_only_t1_in_ovlp_er + num_nt_only_t2_in_ovlp_er;
    let (prop_nt_no_ovlp_in_ovlp_er, prop_nt_ovlp_in_ovlp_er) = if total_nt_in_ovlp_er == 0 {
        (0.0, 0.0)
    } else {
        let no = (num_nt_only_t1_in_ovlp_er + num_nt_only_t2_in_ovlp_er) as f64
            / total_nt_in_ovlp_er as f64;
        (no, 1.0 - no)
    };

    // -------- 3) Flags
    let flag_no_ovlp_nt = if prop_nt_no_ovlp == 1.0 { 1 } else { 0 };
    let flag_fsm = if prop_jxn_no_ovlp == 0.0 { 1 } else { 0 };

    let mut flag_t1_ism_of_t2 = 0;
    let mut flag_t2_ism_of_t1 = 0;
    let has_junctions = !t1.junctions.is_empty() || !t2.junctions.is_empty();
    if has_junctions && flag_fsm == 0 && !jxn_string_t1.is_empty() && !jxn_string_t2.is_empty() {
        if jxn_string_t2.contains(&jxn_string_t1) {
            flag_t1_ism_of_t2 = 1;
        }
        if jxn_string_t1.contains(&jxn_string_t2) {
            flag_t2_ism_of_t1 = 1;
        }
    }

    let flag_ir = if !ir_ef_t1_list.is_empty() || !ir_ef_t2_list.is_empty() {
        1
    } else {
        0
    };

    // End variation by strand
    let (flag_5_var, flag_3_var) = if t1.strand == "+" {
        (
            if t1.min_start != t2.min_start { 1 } else { 0 },
            if t1.max_end != t2.max_end { 1 } else { 0 },
        )
    } else {
        (
            if t1.max_end != t2.max_end { 1 } else { 0 },
            if t1.min_start != t2.min_start { 1 } else { 0 },
        )
    };

    // alt exon / alt donor-acceptor
    let (mut flag_alt_exon, mut flag_alt_da) = if flag_ir == 1 {
        (0, 0)
    } else {
        (
            if prop_er_no_ovlp > 0.0 { 1 } else { 0 },
            if prop_nt_no_ovlp_in_ovlp_er > 0.0 && flag_fsm == 0 {
                1
            } else {
                0
            },
        )
    };
    if flag_no_ovlp_nt == 1 {
        flag_alt_exon = 0;
    }

    PairMetrics {
        // junctions
        num_jxn_only_t1,
        num_jxn_only_t2,
        num_jxn_ovlp,
        prop_jxn_no_ovlp,
        prop_jxn_ovlp,
        jxn_string_t1,
        jxn_string_t2,
        jxn_only_t1,
        jxn_only_t2,
        jxn_same,
        // ER
        num_er_only_t1,
        num_er_only_t2,
        num_er_ovlp,
        prop_er_no_ovlp,
        prop_er_ovlp,
        er_only_t1: er_only_t1_list.join("|"),
        er_only_t2: er_only_t2_list.join("|"),
        er_ovlp: er_ovlp_list.join("|"),
        // EF + exon counts
        num_ef_only_t1,
        num_ef_only_t2,
        num_ef_ovlp,
        prop_ef_no_ovlp,
        prop_ef_ovlp,
        ef_only_t1: ef_only_t1_list.join("|"),
        ef_only_t2: ef_only_t2_list.join("|"),
        ef_ovlp: ef_ovlp_list.join("|"),
        num_exon_only_t1,
        num_exon_only_t2,
        num_exon_ovlp,
        // IR
        num_ir_ef_t1: ir_ef_t1_list.len(),
        num_ir_ef_t2: ir_ef_t2_list.len(),
        ir_ef_t1: ir_ef_t1_list.join("|"),
        ir_ef_t2: ir_ef_t2_list.join("|"),
        // nt
        num_nt_ovlp,
        num_nt_only_t1,
        num_nt_only_t2,
        num_nt_no_ovlp,
        total_nt,
        prop_nt_no_ovlp,
        prop_nt_ovlp,
        num_nt_only_t1_in_ovlp_er,
        num_nt_only_t2_in_ovlp_er,
        num_nt_ovlp_in_ovlp_er,
        total_nt_in_ovlp_er,
        prop_nt_no_ovlp_in_ovlp_er,
        prop_nt_ovlp_in_ovlp_er,
        num_nt_only_t1_in_uniq_er,
        num_nt_only_t2_in_uniq_er,
        // flags
        flag_fsm,
        flag_t1_ism_of_t2,
        flag_t2_ism_of_t1,
        flag_ir,
        flag_5_var,
        flag_3_var,
        flag_alt_da,
        flag_alt_exon,
        flag_no_ovlp_nt,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Read exons from both files
    let rec1 = parse_gtf_exons(&args.gtf1)?;
    let rec2 = parse_gtf_exons(&args.gtf2)?;

    // Index: gene_id -> transcript_id -> (exons[], seqname, strand)
    let mut g1: HashMap<String, HashMap<String, (Vec<(i64, i64)>, String, String)>> = HashMap::new();
    for r in rec1 {
        let e = g1
            .entry(r.gene_id.clone())
            .or_default()
            .entry(r.transcript_id.clone())
            .or_insert_with(|| (Vec::new(), r.seqname.clone(), r.strand.clone()));
        e.0.push((r.start, r.end));
    }
    let mut g2: HashMap<String, HashMap<String, (Vec<(i64, i64)>, String, String)>> = HashMap::new();
    for r in rec2 {
        let e = g2
            .entry(r.gene_id.clone())
            .or_default()
            .entry(r.transcript_id.clone())
            .or_insert_with(|| (Vec::new(), r.seqname.clone(), r.strand.clone()));
        e.0.push((r.start, r.end));
    }

    // Build transcript models
    let mut m1: HashMap<String, HashMap<String, TranscriptModel>> = HashMap::new();
    for (gid, txs) in g1 {
        let mut mm = HashMap::new();
        for (tid, (xs, chr, strand)) in txs {
            mm.insert(tid.clone(), build_transcript_model(gid.clone(), tid, xs, chr, strand));
        }
        m1.insert(gid, mm);
    }
    let mut m2: HashMap<String, HashMap<String, TranscriptModel>> = HashMap::new();
    for (gid, txs) in g2 {
        let mut mm = HashMap::new();
        for (tid, (xs, chr, strand)) in txs {
            mm.insert(tid.clone(), build_transcript_model(gid.clone(), tid, xs, chr, strand));
        }
        m2.insert(gid, mm);
    }

    // Common genes
    let set1: HashSet<&String> = m1.keys().collect();
    let set2: HashSet<&String> = m2.keys().collect();
    let common: Vec<&&String> = set1.intersection(&set2).collect();

    // Output vectors (TranD td_df_cols order)
    let mut col_gene_id = Vec::new();
    let mut col_transcript_1 = Vec::new();
    let mut col_transcript_2 = Vec::new();

    let mut col_num_jxn_only_T1 = Vec::new();
    let mut col_num_jxn_only_T2 = Vec::new();
    let mut col_num_jxn_ovlp = Vec::new();
    let mut col_prop_jxn_noOvlp = Vec::new();
    let mut col_prop_jxn_ovlp = Vec::new();
    let mut col_jxn_string_T1 = Vec::new();
    let mut col_jxn_string_T2 = Vec::new();
    let mut col_jxn_only_T1 = Vec::new();
    let mut col_jxn_only_T2 = Vec::new();
    let mut col_jxn_same = Vec::new();

    let mut col_num_ER_only_T1 = Vec::new();
    let mut col_num_ER_only_T2 = Vec::new();
    let mut col_num_ER_ovlp = Vec::new();
    let mut col_prop_ER_noOvlp = Vec::new();
    let mut col_prop_ER_ovlp = Vec::new();
    let mut col_ER_only_T1 = Vec::new();
    let mut col_ER_only_T2 = Vec::new();
    let mut col_ER_ovlp = Vec::new();

    let mut col_num_EF_only_T1 = Vec::new();
    let mut col_num_EF_only_T2 = Vec::new();
    let mut col_num_EF_ovlp = Vec::new();
    let mut col_prop_EF_noOvlp = Vec::new();
    let mut col_prop_EF_ovlp = Vec::new();
    let mut col_EF_only_T1 = Vec::new();
    let mut col_EF_only_T2 = Vec::new();
    let mut col_EF_ovlp = Vec::new();

    let mut col_num_exon_only_T1 = Vec::new();
    let mut col_num_exon_only_T2 = Vec::new();
    let mut col_num_exon_ovlp = Vec::new();

    let mut col_num_IR_EF_T1 = Vec::new();
    let mut col_num_IR_EF_T2 = Vec::new();
    let mut col_IR_EF_T1 = Vec::new();
    let mut col_IR_EF_T2 = Vec::new();

    let mut col_num_nt_ovlp = Vec::new();
    let mut col_num_nt_only_T1 = Vec::new();
    let mut col_num_nt_only_T2 = Vec::new();
    let mut col_num_nt_noOvlp = Vec::new();
    let mut col_total_nt = Vec::new();
    let mut col_prop_nt_noOvlp = Vec::new();
    let mut col_prop_nt_ovlp = Vec::new();
    let mut col_num_nt_only_T1_in_ovlpER = Vec::new();
    let mut col_num_nt_only_T2_in_ovlpER = Vec::new();
    let mut col_num_nt_ovlp_in_ovlpER = Vec::new();
    let mut col_total_nt_in_ovlpER = Vec::new();
    let mut col_prop_nt_noOvlp_in_ovlpER = Vec::new();
    let mut col_prop_nt_ovlp_in_ovlpER = Vec::new();
    let mut col_num_nt_only_T1_in_uniqER = Vec::new();
    let mut col_num_nt_only_T2_in_uniqER = Vec::new();

    let mut col_flag_FSM = Vec::new();
    let mut col_flag_T1_ISM_of_T2 = Vec::new();
    let mut col_flag_T2_ISM_of_T1 = Vec::new();
    let mut col_flag_IR = Vec::new();
    let mut col_flag_5_var = Vec::new();
    let mut col_flag_3_var = Vec::new();
    let mut col_flag_alt_DA = Vec::new();
    let mut col_flag_alt_exon = Vec::new();
    let mut col_flag_no_ovlp_nt = Vec::new();

    // Diagnostics
    let mut total_nxn: u64 = 0;
    let mut bbox_overlap: u64 = 0;
    let mut bbox_pruned: u64 = 0;

    for &&gid in &common {
        let txs1 = &m1[gid];
        let txs2 = &m2[gid];
        total_nxn += (txs1.len() as u64) * (txs2.len() as u64);

        for (tid1, t1) in txs1 {
            for (tid2, t2) in txs2 {
                // Bounding box overlap (optional prefilter)
                let overlaps = t1.max_end > t2.min_start && t1.min_start < t2.max_end;
                if overlaps {
                    bbox_overlap += 1;
                } else {
                    bbox_pruned += 1;
                }
                if !overlaps && !args.include_zero_overlap {
                    continue;
                }

                let m = compute_pairwise_metrics(t1, t2);

                // Append one row
                col_gene_id.push(gid.clone());
                col_transcript_1.push(tid1.clone());
                col_transcript_2.push(tid2.clone());

                col_num_jxn_only_T1.push(m.num_jxn_only_t1 as u32);
                col_num_jxn_only_T2.push(m.num_jxn_only_t2 as u32);
                col_num_jxn_ovlp.push(m.num_jxn_ovlp as u32);
                col_prop_jxn_noOvlp.push(m.prop_jxn_no_ovlp);
                col_prop_jxn_ovlp.push(m.prop_jxn_ovlp);
                col_jxn_string_T1.push(m.jxn_string_t1);
                col_jxn_string_T2.push(m.jxn_string_t2);
                col_jxn_only_T1.push(m.jxn_only_t1);
                col_jxn_only_T2.push(m.jxn_only_t2);
                col_jxn_same.push(m.jxn_same);

                col_num_ER_only_T1.push(m.num_er_only_t1 as u32);
                col_num_ER_only_T2.push(m.num_er_only_t2 as u32);
                col_num_ER_ovlp.push(m.num_er_ovlp as u32);
                col_prop_ER_noOvlp.push(m.prop_er_no_ovlp);
                col_prop_ER_ovlp.push(m.prop_er_ovlp);
                col_ER_only_T1.push(m.er_only_t1);
                col_ER_only_T2.push(m.er_only_t2);
                col_ER_ovlp.push(m.er_ovlp);

                col_num_EF_only_T1.push(m.num_ef_only_t1 as u32);
                col_num_EF_only_T2.push(m.num_ef_only_t2 as u32);
                col_num_EF_ovlp.push(m.num_ef_ovlp as u32);
                col_prop_EF_noOvlp.push(m.prop_ef_no_ovlp);
                col_prop_EF_ovlp.push(m.prop_ef_ovlp);
                col_EF_only_T1.push(m.ef_only_t1);
                col_EF_only_T2.push(m.ef_only_t2);
                col_EF_ovlp.push(m.ef_ovlp);

                col_num_exon_only_T1.push(m.num_exon_only_t1 as u32);
                col_num_exon_only_T2.push(m.num_exon_only_t2 as u32);
                col_num_exon_ovlp.push(m.num_exon_ovlp as u32);

                col_num_IR_EF_T1.push(m.num_ir_ef_t1 as u32);
                col_num_IR_EF_T2.push(m.num_ir_ef_t2 as u32);
                col_IR_EF_T1.push(m.ir_ef_t1);
                col_IR_EF_T2.push(m.ir_ef_t2);

                col_num_nt_ovlp.push(m.num_nt_ovlp);
                col_num_nt_only_T1.push(m.num_nt_only_t1);
                col_num_nt_only_T2.push(m.num_nt_only_t2);
                col_num_nt_noOvlp.push(m.num_nt_no_ovlp);
                col_total_nt.push(m.total_nt);
                col_prop_nt_noOvlp.push(m.prop_nt_no_ovlp);
                col_prop_nt_ovlp.push(m.prop_nt_ovlp);

                col_num_nt_only_T1_in_ovlpER.push(m.num_nt_only_t1_in_ovlp_er);
                col_num_nt_only_T2_in_ovlpER.push(m.num_nt_only_t2_in_ovlp_er);
                col_num_nt_ovlp_in_ovlpER.push(m.num_nt_ovlp_in_ovlp_er);
                col_total_nt_in_ovlpER.push(m.total_nt_in_ovlp_er);
                col_prop_nt_noOvlp_in_ovlpER.push(m.prop_nt_no_ovlp_in_ovlp_er);
                col_prop_nt_ovlp_in_ovlpER.push(m.prop_nt_ovlp_in_ovlp_er);

                col_num_nt_only_T1_in_uniqER.push(m.num_nt_only_t1_in_uniq_er);
                col_num_nt_only_T2_in_uniqER.push(m.num_nt_only_t2_in_uniq_er);

                col_flag_FSM.push(m.flag_fsm);
                col_flag_T1_ISM_of_T2.push(m.flag_t1_ism_of_t2);
                col_flag_T2_ISM_of_T1.push(m.flag_t2_ism_of_t1);
                col_flag_IR.push(m.flag_ir);
                col_flag_5_var.push(m.flag_5_var);
                col_flag_3_var.push(m.flag_3_var);
                col_flag_alt_DA.push(m.flag_alt_da);
                col_flag_alt_exon.push(m.flag_alt_exon);
                col_flag_no_ovlp_nt.push(m.flag_no_ovlp_nt);
            }
        }
    }

    // Build a Polars DataFrame and write Parquet
    let mut df = DataFrame::new(vec![
        Series::new("gene_id", col_gene_id),
        Series::new("transcript_1", col_transcript_1),
        Series::new("transcript_2", col_transcript_2),
        Series::new("num_jxn_only_T1", col_num_jxn_only_T1),
        Series::new("num_jxn_only_T2", col_num_jxn_only_T2),
        Series::new("num_jxn_ovlp", col_num_jxn_ovlp),
        Series::new("prop_jxn_noOvlp", col_prop_jxn_noOvlp),
        Series::new("prop_jxn_ovlp", col_prop_jxn_ovlp),
        Series::new("jxn_string_T1", col_jxn_string_T1),
        Series::new("jxn_string_T2", col_jxn_string_T2),
        Series::new("jxn_only_T1", col_jxn_only_T1),
        Series::new("jxn_only_T2", col_jxn_only_T2),
        Series::new("jxn_same", col_jxn_same),
        Series::new("num_ER_only_T1", col_num_ER_only_T1),
        Series::new("num_ER_only_T2", col_num_ER_only_T2),
        Series::new("num_ER_ovlp", col_num_ER_ovlp),
        Series::new("prop_ER_noOvlp", col_prop_ER_noOvlp),
        Series::new("prop_ER_ovlp", col_prop_ER_ovlp),
        Series::new("ER_only_T1", col_ER_only_T1),
        Series::new("ER_only_T2", col_ER_only_T2),
        Series::new("ER_ovlp", col_ER_ovlp),
        Series::new("num_EF_only_T1", col_num_EF_only_T1),
        Series::new("num_EF_only_T2", col_num_EF_only_T2),
        Series::new("num_EF_ovlp", col_num_EF_ovlp),
        Series::new("prop_EF_noOvlp", col_prop_EF_noOvlp),
        Series::new("prop_EF_ovlp", col_prop_EF_ovlp),
        Series::new("EF_only_T1", col_EF_only_T1),
        Series::new("EF_only_T2", col_EF_only_T2),
        Series::new("EF_ovlp", col_EF_ovlp),
        Series::new("num_exon_only_T1", col_num_exon_only_T1),
        Series::new("num_exon_only_T2", col_num_exon_only_T2),
        Series::new("num_exon_ovlp", col_num_exon_ovlp),
        Series::new("num_IR_EF_T1", col_num_IR_EF_T1),
        Series::new("num_IR_EF_T2", col_num_IR_EF_T2),
        Series::new("IR_EF_T1", col_IR_EF_T1),
        Series::new("IR_EF_T2", col_IR_EF_T2),
        Series::new("num_nt_ovlp", col_num_nt_ovlp),
        Series::new("num_nt_only_T1", col_num_nt_only_T1),
        Series::new("num_nt_only_T2", col_num_nt_only_T2),
        Series::new("num_nt_noOvlp", col_num_nt_noOvlp),
        Series::new("total_nt", col_total_nt),
        Series::new("prop_nt_noOvlp", col_prop_nt_noOvlp),
        Series::new("prop_nt_ovlp", col_prop_nt_ovlp),
        Series::new("num_nt_only_T1_in_ovlpER", col_num_nt_only_T1_in_ovlpER),
        Series::new("num_nt_only_T2_in_ovlpER", col_num_nt_only_T2_in_ovlpER),
        Series::new("num_nt_ovlp_in_ovlpER", col_num_nt_ovlp_in_ovlpER),
        Series::new("total_nt_in_ovlpER", col_total_nt_in_ovlpER),
        Series::new("prop_nt_noOvlp_in_ovlpER", col_prop_nt_noOvlp_in_ovlpER),
        Series::new("prop_nt_ovlp_in_ovlpER", col_prop_nt_ovlp_in_ovlpER),
        Series::new("num_nt_only_T1_in_uniqER", col_num_nt_only_T1_in_uniqER),
        Series::new("num_nt_only_T2_in_uniqER", col_num_nt_only_T2_in_uniqER),
        Series::new("flag_FSM", col_flag_FSM),
        Series::new("flag_T1_ISM_of_T2", col_flag_T1_ISM_of_T2),
        Series::new("flag_T2_ISM_of_T1", col_flag_T2_ISM_of_T1),
        Series::new("flag_IR", col_flag_IR),
        Series::new("flag_5_var", col_flag_5_var),
        Series::new("flag_3_var", col_flag_3_var),
        Series::new("flag_alt_DA", col_flag_alt_DA),
        Series::new("flag_alt_exon", col_flag_alt_exon),
        Series::new("flag_no_ovlp_nt", col_flag_no_ovlp_nt),
    ])?;

    let file = File::create(&args.out)?;
    ParquetWriter::new(file).finish(&mut df)?;
    println!("Wrote pairwise TranD distances to {:?}", args.out);
    println!(
        "Diagnostics: common genes={}, total pairs={}, bbox overlap={}, bbox pruned={}",
        common.len(),
        total_nxn,
        bbox_overlap,
        bbox_pruned
    );

    Ok(())
}

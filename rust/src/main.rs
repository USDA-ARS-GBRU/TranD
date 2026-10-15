use anyhow::{Context, Result};
use clap::Parser;
use polars::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "TranD Rust GTF Distance Calculator", long_about = None)]
struct Args {
    #[arg(long)]
    gtf1: PathBuf,

    #[arg(long)]
    gtf2: PathBuf,

    #[arg(long, default_value = "transcript_distance.parquet")]
    out: PathBuf,

    /// Include non-overlapping transcript pairs (flag_no_ovlp_nt = 1) in the output table
    #[arg(long, default_value_t = false)]
    include_zero_overlap: bool,
}

#[derive(Debug, Clone)]
struct ExonRecord {
    seqname: String,
    strand: String,
    start: i64,
    end: i64,
    gene_id: String,
    transcript_id: String,
}

#[derive(Debug, Clone)]
struct TranscriptModel {
    seqname: String,
    strand: String,
    gene_id: String,
    transcript_id: String,
    min_start: i64,
    max_end: i64,
    exons: Vec<(i64, i64)>,      // Sorted disjoint intervals
    junctions: Vec<String>,       // Ordered junction strings
    introns: Vec<(i64, i64)>,     // Intron intervals
}

fn parse_gtf_exons(path: &PathBuf) -> Result<Vec<ExonRecord>> {
    let file = File::open(path).with_context(|| format!("Failed to open GTF file: {:?}", path))?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();

    for line_res in reader.lines() {
        let line = line_res?;
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
        let attributes = parts[8];

        let mut gene_id = None;
        let mut transcript_id = None;

        for attr in attributes.split(';') {
            let attr_trim = attr.trim();
            if attr_trim.starts_with("gene_id") {
                if let Some(val) = attr_trim.split_whitespace().nth(1) {
                    gene_id = Some(val.trim_matches('"').to_string());
                }
            } else if attr_trim.starts_with("transcript_id") {
                if let Some(val) = attr_trim.split_whitespace().nth(1) {
                    transcript_id = Some(val.trim_matches('"').to_string());
                }
            }
        }

        if let (Some(gid), Some(tid)) = (gene_id, transcript_id) {
            records.push(ExonRecord {
                seqname,
                strand,
                start,
                end,
                gene_id: gid,
                transcript_id: tid,
            });
        }
    }

    Ok(records)
}

fn merge_intervals(mut intervals: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    if intervals.is_empty() {
        return Vec::new();
    }
    intervals.sort_by_key(|x| x.0);
    let mut merged = vec![intervals[0]];
    for curr in intervals.into_iter().skip(1) {
        let last_idx = merged.len() - 1;
        if curr.0 <= merged[last_idx].1 {
            merged[last_idx].1 = merged[last_idx].1.max(curr.1);
        } else {
            merged.push(curr);
        }
    }
    merged
}

fn build_transcript_model(
    gene_id: String,
    transcript_id: String,
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
        let jct_str = format!("{}:{}:{}:{}", seqname, left_end, right_start, strand);
        junctions.push(jct_str);
        introns.push((exons[i].1, exons[i + 1].0));
    }

    TranscriptModel {
        seqname,
        strand,
        gene_id,
        transcript_id,
        min_start,
        max_end,
        exons,
        junctions,
        introns,
    }
}

struct PairMetrics {
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

    num_er_only_t1: usize,
    num_er_only_t2: usize,
    num_er_ovlp: usize,
    prop_er_no_ovlp: f64,
    prop_er_ovlp: f64,
    er_only_t1: String,
    er_only_t2: String,
    er_ovlp: String,

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

    num_ir_ef_t1: usize,
    num_ir_ef_t2: usize,
    ir_ef_t1: String,
    ir_ef_t2: String,

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

fn compute_pairwise_metrics(t1: &TranscriptModel, t2: &TranscriptModel) -> PairMetrics {
    // 1. Junction metrics
    let jxn_string_t1 = t1.junctions.join("|");
    let jxn_string_t2 = t2.junctions.join("|");

    let set_j1: HashSet<&String> = t1.junctions.iter().collect();
    let set_j2: HashSet<&String> = t2.junctions.iter().collect();

    let shared_jxns: Vec<&String> = t1.junctions.iter().filter(|j| set_j2.contains(j)).collect();
    let only_t1_jxns: Vec<&String> = t1.junctions.iter().filter(|j| !set_j2.contains(j)).collect();
    let only_t2_jxns: Vec<&String> = t2.junctions.iter().filter(|j| !set_j1.contains(j)).collect();

    let num_jxn_ovlp = shared_jxns.len();
    let num_jxn_only_t1 = only_t1_jxns.len();
    let num_jxn_only_t2 = only_t2_jxns.len();
    let total_jxn_union = num_jxn_ovlp + num_jxn_only_t1 + num_jxn_only_t2;

    let (prop_jxn_no_ovlp, prop_jxn_ovlp) = if total_jxn_union == 0 {
        (0.0, 1.0)
    } else {
        let no_ovlp = ((num_jxn_only_t1 + num_jxn_only_t2) as f64) / (total_jxn_union as f64);
        (no_ovlp, 1.0 - no_ovlp)
    };

    let jxn_only_t1 = only_t1_jxns.into_iter().cloned().collect::<Vec<_>>().join("|");
    let jxn_only_t2 = only_t2_jxns.into_iter().cloned().collect::<Vec<_>>().join("|");
    let jxn_same = shared_jxns.into_iter().cloned().collect::<Vec<_>>().join("|");

    // 2. Exonic Regions (ER) and Exonic Fragments (EF)
    // Combine exons to generate merged Exonic Regions (ER)
    let mut all_exons = t1.exons.clone();
    all_exons.extend(t2.exons.clone());
    let ers = merge_intervals(all_exons);

    // Identify EF boundaries
    let mut boundaries = Vec::new();
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

    #[derive(Debug)]
    struct EfInfo {
        id: String,
        er_id: String,
        start: i64,
        end: i64,
        in_t1: bool,
        in_t2: bool,
        is_ir_t1: bool,
        is_ir_t2: bool,
    }

    let is_in_exons = |start: i64, end: i64, exons: &[(i64, i64)]| -> bool {
        exons.iter().any(|&(es, ee)| es <= start && end <= ee)
    };

    let is_in_introns = |start: i64, end: i64, introns: &[(i64, i64)]| -> bool {
        introns.iter().any(|&(is, ie)| start >= is && end <= ie)
    };

    let mut ef_list = Vec::new();
    let mut er_ef_counts: HashMap<String, usize> = HashMap::new();
    let mut er_t1_overlap: HashMap<String, bool> = HashMap::new();
    let mut er_t2_overlap: HashMap<String, bool> = HashMap::new();

    for er_idx in 0..ers.len() {
        let (er_start, er_end) = ers[er_idx];
        let er_id = format!("{}:{}:{}:{}", t1.seqname, er_start, er_end, t1.strand);
        let mut ef_idx = 1;

        for i in 0..boundaries.len().saturating_sub(1) {
            let b_start = boundaries[i];
            let b_end = boundaries[i + 1];

            if b_start >= er_start && b_end <= er_end && b_start < b_end {
                let in_t1 = is_in_exons(b_start, b_end, &t1.exons);
                let in_t2 = is_in_exons(b_start, b_end, &t2.exons);

                if in_t1 || in_t2 {
                    let ef_id = format!("{}:{}:{}:{}", t1.seqname, b_start, b_end, t1.strand);

                    // Check Intron Retention (IR)
                    let is_ir_t1 = in_t1 && is_in_introns(b_start, b_end, &t2.introns);
                    let is_ir_t2 = in_t2 && is_in_introns(b_start, b_end, &t1.introns);

                    ef_list.push(EfInfo {
                        id: ef_id,
                        er_id: er_id.clone(),
                        start: b_start,
                        end: b_end,
                        in_t1,
                        in_t2,
                        is_ir_t1,
                        is_ir_t2,
                    });

                    *er_ef_counts.entry(er_id.clone()).or_default() += 1;
                    if in_t1 {
                        er_t1_overlap.insert(er_id.clone(), true);
                    }
                    if in_t2 {
                        er_t2_overlap.insert(er_id.clone(), true);
                    }
                    ef_idx += 1;
                }
            }
        }
    }

    // ER distance summaries
    let mut er_only_t1_list = Vec::new();
    let mut er_only_t2_list = Vec::new();
    let mut er_ovlp_list = Vec::new();

    for (er_start, er_end) in &ers {
        let er_id = format!("{}:{}:{}:{}", t1.seqname, er_start, er_end, t1.strand);
        let has_t1 = er_t1_overlap.get(&er_id).copied().unwrap_or(false);
        let has_t2 = er_t2_overlap.get(&er_id).copied().unwrap_or(false);

        if has_t1 && has_t2 {
            er_ovlp_list.push(er_id);
        } else if has_t1 {
            er_only_t1_list.push(er_id);
        } else if has_t2 {
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
        let no_ovlp = ((num_er_only_t1 + num_er_only_t2) as f64) / (total_er_union as f64);
        (no_ovlp, 1.0 - no_ovlp)
    };

    let shared_er_set: HashSet<&String> = er_ovlp_list.iter().collect();

    // EF distance & nucleotide summaries
    let mut ef_only_t1_list = Vec::new();
    let mut ef_only_t2_list = Vec::new();
    let mut ef_ovlp_list = Vec::new();

    let mut num_exon_only_t1 = 0;
    let mut num_exon_only_t2 = 0;
    let mut num_exon_ovlp = 0;

    let mut ir_ef_t1_list = Vec::new();
    let mut ir_ef_t2_list = Vec::new();

    let mut num_nt_ovlp = 0i64;
    let mut num_nt_only_t1 = 0i64;
    let mut num_nt_only_t2 = 0i64;

    let mut num_nt_only_t1_in_ovlp_er = 0i64;
    let mut num_nt_only_t2_in_ovlp_er = 0i64;
    let mut num_nt_ovlp_in_ovlp_er = 0i64;

    let mut num_nt_only_t1_in_uniq_er = 0i64;
    let mut num_nt_only_t2_in_uniq_er = 0i64;

    for ef in &ef_list {
        let length = ef.end - ef.start;
        let is_singleton = er_ef_counts.get(&ef.er_id).copied().unwrap_or(0) == 1;
        let in_shared_er = shared_er_set.contains(&ef.er_id);

        if ef.is_ir_t1 {
            ir_ef_t1_list.push(ef.id.clone());
        }
        if ef.is_ir_t2 {
            ir_ef_t2_list.push(ef.id.clone());
        }

        if ef.in_t1 && ef.in_t2 {
            ef_ovlp_list.push(ef.id.clone());
            if is_singleton {
                num_exon_ovlp += 1;
            }
            num_nt_ovlp += length;
            if in_shared_er {
                num_nt_ovlp_in_ovlp_er += length;
            }
        } else if ef.in_t1 {
            ef_only_t1_list.push(ef.id.clone());
            if is_singleton {
                num_exon_only_t1 += 1;
            }
            num_nt_only_t1 += length;
            if in_shared_er {
                num_nt_only_t1_in_ovlp_er += length;
            } else {
                num_nt_only_t1_in_uniq_er += length;
            }
        } else if ef.in_t2 {
            ef_only_t2_list.push(ef.id.clone());
            if is_singleton {
                num_exon_only_t2 += 1;
            }
            num_nt_only_t2 += length;
            if in_shared_er {
                num_nt_only_t2_in_ovlp_er += length;
            } else {
                num_nt_only_t2_in_uniq_er += length;
            }
        }
    }

    let num_ef_only_t1 = ef_only_t1_list.len();
    let num_ef_only_t2 = ef_only_t2_list.len();
    let num_ef_ovlp = ef_ovlp_list.len();
    let total_ef_union = num_ef_only_t1 + num_ef_only_t2 + num_ef_ovlp;

    let (prop_ef_no_ovlp, prop_ef_ovlp) = if total_ef_union == 0 {
        (0.0, 1.0)
    } else {
        let no_ovlp = ((num_ef_only_t1 + num_ef_only_t2) as f64) / (total_ef_union as f64);
        (no_ovlp, 1.0 - no_ovlp)
    };

    let num_nt_no_ovlp = num_nt_only_t1 + num_nt_only_t2;
    let total_nt = num_nt_ovlp + num_nt_no_ovlp;

    let (prop_nt_no_ovlp, prop_nt_ovlp) = if total_nt == 0 {
        (0.0, 1.0)
    } else {
        let no_ovlp = (num_nt_no_ovlp as f64) / (total_nt as f64);
        (no_ovlp, 1.0 - no_ovlp)
    };

    let total_nt_in_ovlp_er = num_nt_ovlp_in_ovlp_er + num_nt_only_t1_in_ovlp_er + num_nt_only_t2_in_ovlp_er;
    let (prop_nt_no_ovlp_in_ovlp_er, prop_nt_ovlp_in_ovlp_er) = if total_nt_in_ovlp_er == 0 {
        (0.0, 0.0)
    } else {
        let no_ovlp = ((num_nt_only_t1_in_ovlp_er + num_nt_only_t2_in_ovlp_er) as f64) / (total_nt_in_ovlp_er as f64);
        (no_ovlp, 1.0 - no_ovlp)
    };

    // 3. Splicing Pattern Flags
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

    let flag_ir = if ir_ef_t1_list.len() + ir_ef_t2_list.len() > 0 { 1 } else { 0 };

    let t1_min_start = t1.min_start;
    let t2_min_start = t2.min_start;
    let t1_max_end = t1.max_end;
    let t2_max_end = t2.max_end;

    let (flag_5_var, flag_3_var) = if t1.strand == "+" {
        (
            if t1_min_start != t2_min_start { 1 } else { 0 },
            if t1_max_end != t2_max_end { 1 } else { 0 },
        )
    } else {
        (
            if t1_max_end != t2_max_end { 1 } else { 0 },
            if t1_min_start != t2_min_start { 1 } else { 0 },
        )
    };

    let (mut flag_alt_exon, mut flag_alt_da) = if flag_ir == 1 {
        (0, 0)
    } else {
        (
            if prop_er_no_ovlp > 0.0 { 1 } else { 0 },
            if prop_nt_no_ovlp_in_ovlp_er > 0.0 && flag_fsm == 0 { 1 } else { 0 },
        )
    };

    if flag_no_ovlp_nt == 1 {
        flag_alt_exon = 0;
    }

    PairMetrics {
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

        num_er_only_t1,
        num_er_only_t2,
        num_er_ovlp,
        prop_er_no_ovlp,
        prop_er_ovlp,
        er_only_t1: er_only_t1_list.join("|"),
        er_only_t2: er_only_t2_list.join("|"),
        er_ovlp: er_ovlp_list.join("|"),

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

        num_ir_ef_t1: ir_ef_t1_list.len(),
        num_ir_ef_t2: ir_ef_t2_list.len(),
        ir_ef_t1: ir_ef_t1_list.join("|"),
        ir_ef_t2: ir_ef_t2_list.join("|"),

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

    println!("Parsing GTF1: {:?}", args.gtf1);
    let records1 = parse_gtf_exons(&args.gtf1)?;
    println!("Parsing GTF2: {:?}", args.gtf2);
    let records2 = parse_gtf_exons(&args.gtf2)?;

    // Map gene_id -> transcript_id -> Exon records
    let mut gene_tx_map1: HashMap<String, HashMap<String, (Vec<(i64, i64)>, String, String)>> = HashMap::new();
    for rec in records1 {
        let entry = gene_tx_map1
            .entry(rec.gene_id.clone())
            .or_default()
            .entry(rec.transcript_id.clone())
            .or_insert_with(|| (Vec::new(), rec.seqname.clone(), rec.strand.clone()));
        entry.0.push((rec.start, rec.end));
    }

    let mut gene_tx_map2: HashMap<String, HashMap<String, (Vec<(i64, i64)>, String, String)>> = HashMap::new();
    for rec in records2 {
        let entry = gene_tx_map2
            .entry(rec.gene_id.clone())
            .or_default()
            .entry(rec.transcript_id.clone())
            .or_insert_with(|| (Vec::new(), rec.seqname.clone(), rec.strand.clone()));
        entry.0.push((rec.start, rec.end));
    }

    let mut gene_models1: HashMap<String, HashMap<String, TranscriptModel>> = HashMap::new();
    for (gid, tx_map) in gene_tx_map1 {
        let mut model_map = HashMap::new();
        for (tid, (exons, seqname, strand)) in tx_map {
            let model = build_transcript_model(gid.clone(), tid.clone(), exons, seqname, strand);
            model_map.insert(tid, model);
        }
        gene_models1.insert(gid, model_map);
    }

    let mut gene_models2: HashMap<String, HashMap<String, TranscriptModel>> = HashMap::new();
    for (gid, tx_map) in gene_tx_map2 {
        let mut model_map = HashMap::new();
        for (tid, (exons, seqname, strand)) in tx_map {
            let model = build_transcript_model(gid.clone(), tid.clone(), exons, seqname, strand);
            model_map.insert(tid, model);
        }
        gene_models2.insert(gid, model_map);
    }

    let set1: HashSet<&String> = gene_models1.keys().collect();
    let set2: HashSet<&String> = gene_models2.keys().collect();
    let common_genes: Vec<&&String> = set1.intersection(&set2).collect();

    // Data structures for full TranD transcript distance columns
    let mut col_gene_id = Vec::new();
    let mut col_transcript_1 = Vec::new();
    let mut col_transcript_2 = Vec::new();

    let mut col_num_jxn_only_t1 = Vec::new();
    let mut col_num_jxn_only_t2 = Vec::new();
    let mut col_num_jxn_ovlp = Vec::new();
    let mut col_prop_jxn_no_ovlp = Vec::new();
    let mut col_prop_jxn_ovlp = Vec::new();
    let mut col_jxn_string_t1 = Vec::new();
    let mut col_jxn_string_t2 = Vec::new();
    let mut col_jxn_only_t1 = Vec::new();
    let mut col_jxn_only_t2 = Vec::new();
    let mut col_jxn_same = Vec::new();

    let mut col_num_er_only_t1 = Vec::new();
    let mut col_num_er_only_t2 = Vec::new();
    let mut col_num_er_ovlp = Vec::new();
    let mut col_prop_er_no_ovlp = Vec::new();
    let mut col_prop_er_ovlp = Vec::new();
    let mut col_er_only_t1 = Vec::new();
    let mut col_er_only_t2 = Vec::new();
    let mut col_er_ovlp = Vec::new();

    let mut col_num_ef_only_t1 = Vec::new();
    let mut col_num_ef_only_t2 = Vec::new();
    let mut col_num_ef_ovlp = Vec::new();
    let mut col_prop_ef_no_ovlp = Vec::new();
    let mut col_prop_ef_ovlp = Vec::new();
    let mut col_ef_only_t1 = Vec::new();
    let mut col_ef_only_t2 = Vec::new();
    let mut col_ef_ovlp = Vec::new();

    let mut col_num_exon_only_t1 = Vec::new();
    let mut col_num_exon_only_t2 = Vec::new();
    let mut col_num_exon_ovlp = Vec::new();

    let mut col_num_ir_ef_t1 = Vec::new();
    let mut col_num_ir_ef_t2 = Vec::new();
    let mut col_ir_ef_t1 = Vec::new();
    let mut col_ir_ef_t2 = Vec::new();

    let mut col_num_nt_ovlp = Vec::new();
    let mut col_num_nt_only_t1 = Vec::new();
    let mut col_num_nt_only_t2 = Vec::new();
    let mut col_num_nt_no_ovlp = Vec::new();
    let mut col_total_nt = Vec::new();
    let mut col_prop_nt_no_ovlp = Vec::new();
    let mut col_prop_nt_ovlp = Vec::new();

    let mut col_num_nt_only_t1_in_ovlp_er = Vec::new();
    let mut col_num_nt_only_t2_in_ovlp_er = Vec::new();
    let mut col_num_nt_ovlp_in_ovlp_er = Vec::new();
    let mut col_total_nt_in_ovlp_er = Vec::new();
    let mut col_prop_nt_no_ovlp_in_ovlp_er = Vec::new();
    let mut col_prop_nt_ovlp_in_ovlp_er = Vec::new();

    let mut col_num_nt_only_t1_in_uniq_er = Vec::new();
    let mut col_num_nt_only_t2_in_uniq_er = Vec::new();

    let mut col_flag_fsm = Vec::new();
    let mut col_flag_t1_ism_of_t2 = Vec::new();
    let mut col_flag_t2_ism_of_t1 = Vec::new();
    let mut col_flag_ir = Vec::new();
    let mut col_flag_5_var = Vec::new();
    let mut col_flag_3_var = Vec::new();
    let mut col_flag_alt_da = Vec::new();
    let mut col_flag_alt_exon = Vec::new();
    let mut col_flag_no_ovlp_nt = Vec::new();

    let mut total_nxn_pairs: u64 = 0;
    let mut bbox_overlapping_pairs: u64 = 0;
    let mut bbox_disjoint_pruned_pairs: u64 = 0;

    for &&gene_id in &common_genes {
        let txs1 = &gene_models1[gene_id];
        let txs2 = &gene_models2[gene_id];

        let n1 = txs1.len() as u64;
        let n2 = txs2.len() as u64;
        total_nxn_pairs += n1 * n2;

        for (tid1, model1) in txs1 {
            for (tid2, model2) in txs2 {
                let bbox_overlaps = model1.max_end > model2.min_start && model1.min_start < model2.max_end;

                if bbox_overlaps {
                    bbox_overlapping_pairs += 1;
                } else {
                    bbox_disjoint_pruned_pairs += 1;
                }

                if bbox_overlaps || args.include_zero_overlap {
                    let m = compute_pairwise_metrics(model1, model2);

                    col_gene_id.push(gene_id.clone());
                    col_transcript_1.push(tid1.clone());
                    col_transcript_2.push(tid2.clone());

                    col_num_jxn_only_t1.push(m.num_jxn_only_t1 as u32);
                    col_num_jxn_only_t2.push(m.num_jxn_only_t2 as u32);
                    col_num_jxn_ovlp.push(m.num_jxn_ovlp as u32);
                    col_prop_jxn_no_ovlp.push(m.prop_jxn_no_ovlp);
                    col_prop_jxn_ovlp.push(m.prop_jxn_ovlp);
                    col_jxn_string_t1.push(m.jxn_string_t1);
                    col_jxn_string_t2.push(m.jxn_string_t2);
                    col_jxn_only_t1.push(m.jxn_only_t1);
                    col_jxn_only_t2.push(m.jxn_only_t2);
                    col_jxn_same.push(m.jxn_same);

                    col_num_er_only_t1.push(m.num_er_only_t1 as u32);
                    col_num_er_only_t2.push(m.num_er_only_t2 as u32);
                    col_num_er_ovlp.push(m.num_er_ovlp as u32);
                    col_prop_er_no_ovlp.push(m.prop_er_no_ovlp);
                    col_prop_er_ovlp.push(m.prop_er_ovlp);
                    col_er_only_t1.push(m.er_only_t1);
                    col_er_only_t2.push(m.er_only_t2);
                    col_er_ovlp.push(m.er_ovlp);

                    col_num_ef_only_t1.push(m.num_ef_only_t1 as u32);
                    col_num_ef_only_t2.push(m.num_ef_only_t2 as u32);
                    col_num_ef_ovlp.push(m.num_ef_ovlp as u32);
                    col_prop_ef_no_ovlp.push(m.prop_ef_no_ovlp);
                    col_prop_ef_ovlp.push(m.prop_ef_ovlp);
                    col_ef_only_t1.push(m.ef_only_t1);
                    col_ef_only_t2.push(m.ef_only_t2);
                    col_ef_ovlp.push(m.ef_ovlp);

                    col_num_exon_only_t1.push(m.num_exon_only_t1 as u32);
                    col_num_exon_only_t2.push(m.num_exon_only_t2 as u32);
                    col_num_exon_ovlp.push(m.num_exon_ovlp as u32);

                    col_num_ir_ef_t1.push(m.num_ir_ef_t1 as u32);
                    col_num_ir_ef_t2.push(m.num_ir_ef_t2 as u32);
                    col_ir_ef_t1.push(m.ir_ef_t1);
                    col_ir_ef_t2.push(m.ir_ef_t2);

                    col_num_nt_ovlp.push(m.num_nt_ovlp);
                    col_num_nt_only_t1.push(m.num_nt_only_t1);
                    col_num_nt_only_t2.push(m.num_nt_only_t2);
                    col_num_nt_no_ovlp.push(m.num_nt_no_ovlp);
                    col_total_nt.push(m.total_nt);
                    col_prop_nt_no_ovlp.push(m.prop_nt_no_ovlp);
                    col_prop_nt_ovlp.push(m.prop_nt_ovlp);

                    col_num_nt_only_t1_in_ovlp_er.push(m.num_nt_only_t1_in_ovlp_er);
                    col_num_nt_only_t2_in_ovlp_er.push(m.num_nt_only_t2_in_ovlp_er);
                    col_num_nt_ovlp_in_ovlp_er.push(m.num_nt_ovlp_in_ovlp_er);
                    col_total_nt_in_ovlp_er.push(m.total_nt_in_ovlp_er);
                    col_prop_nt_no_ovlp_in_ovlp_er.push(m.prop_nt_no_ovlp_in_ovlp_er);
                    col_prop_nt_ovlp_in_ovlp_er.push(m.prop_nt_ovlp_in_ovlp_er);

                    col_num_nt_only_t1_in_uniq_er.push(m.num_nt_only_t1_in_uniq_er);
                    col_num_nt_only_t2_in_uniq_er.push(m.num_nt_only_t2_in_uniq_er);

                    col_flag_fsm.push(m.flag_fsm);
                    col_flag_t1_ism_of_t2.push(m.flag_t1_ism_of_t2);
                    col_flag_t2_ism_of_t1.push(m.flag_t2_ism_of_t1);
                    col_flag_ir.push(m.flag_ir);
                    col_flag_5_var.push(m.flag_5_var);
                    col_flag_3_var.push(m.flag_3_var);
                    col_flag_alt_da.push(m.flag_alt_da);
                    col_flag_alt_exon.push(m.flag_alt_exon);
                    col_flag_no_ovlp_nt.push(m.flag_no_ovlp_nt);
                }
            }
        }
    }

    let total_output_rows = col_gene_id.len();

    println!("=== Diagnostic & Processing Summary ===");
    println!("Total common genes processed:                       {}", common_genes.len());
    println!("Total NxN candidate transcript pairs within genes:   {}", total_nxn_pairs);
    println!("Bounding-box overlapping pairs evaluated:            {}", bbox_overlapping_pairs);
    println!("Bounding-box disjoint pairs pruned:                  {}", bbox_disjoint_pruned_pairs);
    println!("Total rows written to output file:                  {}", total_output_rows);
    println!("=======================================");

    let mut out_df = DataFrame::new(vec![
        Series::new("gene_id", col_gene_id),
        Series::new("transcript_1", col_transcript_1),
        Series::new("transcript_2", col_transcript_2),
        Series::new("num_jxn_only_T1", col_num_jxn_only_t1),
        Series::new("num_jxn_only_T2", col_num_jxn_only_t2),
        Series::new("num_jxn_ovlp", col_num_jxn_ovlp),
        Series::new("prop_jxn_noOvlp", col_prop_jxn_no_ovlp),
        Series::new("prop_jxn_ovlp", col_prop_jxn_ovlp),
        Series::new("jxn_string_T1", col_jxn_string_t1),
        Series::new("jxn_string_T2", col_jxn_string_t2),
        Series::new("jxn_only_T1", col_jxn_only_t1),
        Series::new("jxn_only_T2", col_jxn_only_t2),
        Series::new("jxn_same", col_jxn_same),
        Series::new("num_ER_only_T1", col_num_er_only_t1),
        Series::new("num_ER_only_T2", col_num_er_only_t2),
        Series::new("num_ER_ovlp", col_num_er_ovlp),
        Series::new("prop_ER_noOvlp", col_prop_er_no_ovlp),
        Series::new("prop_ER_ovlp", col_prop_er_ovlp),
        Series::new("ER_only_T1", col_er_only_t1),
        Series::new("ER_only_T2", col_er_only_t2),
        Series::new("ER_ovlp", col_er_ovlp),
        Series::new("num_EF_only_T1", col_num_ef_only_t1),
        Series::new("num_EF_only_T2", col_num_ef_only_t2),
        Series::new("num_EF_ovlp", col_num_ef_ovlp),
        Series::new("prop_EF_noOvlp", col_prop_ef_no_ovlp),
        Series::new("prop_EF_ovlp", col_prop_ef_ovlp),
        Series::new("EF_only_T1", col_ef_only_t1),
        Series::new("EF_only_T2", col_ef_only_t2),
        Series::new("EF_ovlp", col_ef_ovlp),
        Series::new("num_exon_only_T1", col_num_exon_only_t1),
        Series::new("num_exon_only_T2", col_num_exon_only_t2),
        Series::new("num_exon_ovlp", col_num_exon_ovlp),
        Series::new("num_IR_EF_T1", col_num_ir_ef_t1),
        Series::new("num_IR_EF_T2", col_num_ir_ef_t2),
        Series::new("IR_EF_T1", col_ir_ef_t1),
        Series::new("IR_EF_T2", col_ir_ef_t2),
        Series::new("num_nt_ovlp", col_num_nt_ovlp),
        Series::new("num_nt_only_T1", col_num_nt_only_t1),
        Series::new("num_nt_only_T2", col_num_nt_only_t2),
        Series::new("num_nt_noOvlp", col_num_nt_no_ovlp),
        Series::new("total_nt", col_total_nt),
        Series::new("prop_nt_noOvlp", col_prop_nt_no_ovlp),
        Series::new("prop_nt_ovlp", col_prop_nt_ovlp),
        Series::new("num_nt_only_T1_in_ovlpER", col_num_nt_only_t1_in_ovlp_er),
        Series::new("num_nt_only_T2_in_ovlpER", col_num_nt_only_t2_in_ovlp_er),
        Series::new("num_nt_ovlp_in_ovlpER", col_num_nt_ovlp_in_ovlp_er),
        Series::new("total_nt_in_ovlpER", col_total_nt_in_ovlp_er),
        Series::new("prop_nt_noOvlp_in_ovlpER", col_prop_nt_no_ovlp_in_ovlp_er),
        Series::new("prop_nt_ovlp_in_ovlpER", col_prop_nt_ovlp_in_ovlp_er),
        Series::new("num_nt_only_T1_in_uniqER", col_num_nt_only_t1_in_uniq_er),
        Series::new("num_nt_only_T2_in_uniqER", col_num_nt_only_t2_in_uniq_er),
        Series::new("flag_FSM", col_flag_fsm),
        Series::new("flag_T1_ISM_of_T2", col_flag_t1_ism_of_t2),
        Series::new("flag_T2_ISM_of_T1", col_flag_t2_ism_of_t1),
        Series::new("flag_IR", col_flag_ir),
        Series::new("flag_5_var", col_flag_5_var),
        Series::new("flag_3_var", col_flag_3_var),
        Series::new("flag_alt_DA", col_flag_alt_da),
        Series::new("flag_alt_exon", col_flag_alt_exon),
        Series::new("flag_no_ovlp_nt", col_flag_no_ovlp_nt),
    ])?;

    println!("Writing Parquet output to: {:?}", args.out);
    let file = File::create(&args.out)?;
    ParquetWriter::new(file).finish(&mut out_df)?;

    println!("Successfully wrote full TranD transcript distance metrics to Parquet file!");
    Ok(())
}

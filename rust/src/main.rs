use anyhow::{Context, Result};
use clap::Parser;
use polars::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "TranD Rust GTF Jaccard Distance Calculator", long_about = None)]
struct Args {
    #[arg(long)]
    gtf1: PathBuf,

    #[arg(long)]
    gtf2: PathBuf,

    #[arg(long, default_value = "jaccard.parquet")]
    out: PathBuf,
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

fn parse_gtf_exons(path: &PathBuf) -> Result<(Vec<ExonRecord>, DataFrame)> {
    let file = File::open(path).with_context(|| format!("Failed to open GTF file: {:?}", path))?;
    let reader = BufReader::new(file);

    let mut seqnames = Vec::new();
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    let mut strands = Vec::new();
    let mut gene_ids = Vec::new();
    let mut transcript_ids = Vec::new();

    let mut records = Vec::new();

    for line_res in reader.lines() {
        let line = line_res?;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 9 {
            continue;
        }

        let feature = parts[2];
        if feature != "exon" {
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
            seqnames.push(seqname.clone());
            starts.push(start);
            ends.push(end);
            strands.push(strand.clone());
            gene_ids.push(gid.clone());
            transcript_ids.push(tid.clone());

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

    let df = DataFrame::new(vec![
        Series::new("seqname", seqnames),
        Series::new("start", starts),
        Series::new("end", ends),
        Series::new("strand", strands),
        Series::new("gene_id", gene_ids),
        Series::new("transcript_id", transcript_ids),
    ])?;

    Ok((records, df))
}

#[derive(Debug, Clone)]
struct TranscriptModel {
    min_start: i64,
    max_end: i64,
    exons: Vec<(i64, i64)>, // Sorted disjoint intervals
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

fn calculate_jaccard_distance(exons1: &[(i64, i64)], exons2: &[(i64, i64)]) -> f64 {
    let mut intersection_len: i64 = 0;
    let mut i = 0;
    let mut j = 0;

    while i < exons1.len() && j < exons2.len() {
        let (s1, e1) = exons1[i];
        let (s2, e2) = exons2[j];

        let ovlp_start = s1.max(s2);
        let ovlp_end = e1.min(e2);

        if ovlp_start < ovlp_end {
            intersection_len += ovlp_end - ovlp_start;
        }

        if e1 < e2 {
            i += 1;
        } else {
            j += 1;
        }
    }

    let len1: i64 = exons1.iter().map(|(s, e)| e - s).sum();
    let len2: i64 = exons2.iter().map(|(s, e)| e - s).sum();
    let union_len = len1 + len2 - intersection_len;

    if union_len == 0 {
        0.0
    } else {
        let jaccard_index = (intersection_len as f64) / (union_len as f64);
        1.0 - jaccard_index
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Parsing GTF1: {:?}", args.gtf1);
    let (records1, _df1) = parse_gtf_exons(&args.gtf1)?;
    println!("Parsing GTF2: {:?}", args.gtf2);
    let (records2, _df2) = parse_gtf_exons(&args.gtf2)?;

    // Map gene_id -> transcript_id -> Exon intervals
    let mut gene_tx_map1: HashMap<String, HashMap<String, Vec<(i64, i64)>>> = HashMap::new();
    for rec in records1 {
        gene_tx_map1
            .entry(rec.gene_id)
            .or_default()
            .entry(rec.transcript_id)
            .or_default()
            .push((rec.start, rec.end));
    }

    let mut gene_tx_map2: HashMap<String, HashMap<String, Vec<(i64, i64)>>> = HashMap::new();
    for rec in records2 {
        gene_tx_map2
            .entry(rec.gene_id)
            .or_default()
            .entry(rec.transcript_id)
            .or_default()
            .push((rec.start, rec.end));
    }

    // Convert raw exons to TranscriptModel (merged intervals + bounding boxes)
    let build_tx_models = |raw_map: HashMap<String, Vec<(i64, i64)>>| -> HashMap<String, TranscriptModel> {
        let mut model_map = HashMap::new();
        for (tid, raw_exons) in raw_map {
            let merged = merge_intervals(raw_exons);
            let min_start = merged.iter().map(|x| x.0).min().unwrap_or(0);
            let max_end = merged.iter().map(|x| x.1).max().unwrap_or(0);
            model_map.insert(tid, TranscriptModel { min_start, max_end, exons: merged });
        }
        model_map
    };

    let mut gene_models1: HashMap<String, HashMap<String, TranscriptModel>> = HashMap::new();
    for (gid, tx_map) in gene_tx_map1 {
        gene_models1.insert(gid, build_tx_models(tx_map));
    }

    let mut gene_models2: HashMap<String, HashMap<String, TranscriptModel>> = HashMap::new();
    for (gid, tx_map) in gene_tx_map2 {
        gene_models2.insert(gid, build_tx_models(tx_map));
    }

    let set1: HashSet<&String> = gene_models1.keys().collect();
    let set2: HashSet<&String> = gene_models2.keys().collect();
    let common_genes: Vec<&&String> = set1.intersection(&set2).collect();

    let mut res_gene_ids = Vec::new();
    let mut res_t1_ids = Vec::new();
    let mut res_t2_ids = Vec::new();
    let mut res_jaccard_dists = Vec::new();

    let mut total_nxn_pairs: u64 = 0;
    let mut total_compared_pairs: u64 = 0;

    for &&gene_id in &common_genes {
        let txs1 = &gene_models1[gene_id];
        let txs2 = &gene_models2[gene_id];

        let n1 = txs1.len() as u64;
        let n2 = txs2.len() as u64;
        total_nxn_pairs += n1 * n2;

        for (tid1, model1) in txs1 {
            for (tid2, model2) in txs2 {
                total_compared_pairs += 1;
                // Check if bounding coordinates overlap
                if model1.max_end > model2.min_start && model1.min_start < model2.max_end {
                    let jaccard_dist = calculate_jaccard_distance(&model1.exons, &model2.exons);
                    res_gene_ids.push(gene_id.clone());
                    res_t1_ids.push(tid1.clone());
                    res_t2_ids.push(tid2.clone());
                    res_jaccard_dists.push(jaccard_dist);
                }
            }
        }
    }

    let total_output_rows = res_jaccard_dists.len();

    println!("=== Processing Summary ===");
    println!("Total common genes processed: {}", common_genes.len());
    println!("Sum of NxN transcript pairs across all common genes: {}", total_nxn_pairs);
    println!("Total transcript pairs compared: {}", total_compared_pairs);
    println!("Total rows written to output file: {}", total_output_rows);
    println!("==========================");

    let mut out_df = DataFrame::new(vec![
        Series::new("gene_id", res_gene_ids),
        Series::new("gtf1_transcript_id", res_t1_ids),
        Series::new("gtf2_transcript_id", res_t2_ids),
        Series::new("jaccard_distance", res_jaccard_dists),
    ])?;

    println!("Writing Parquet output to: {:?}", args.out);
    let file = File::create(&args.out)?;
    ParquetWriter::new(file).finish(&mut out_df)?;

    println!("Successfully wrote Jaccard distances to Parquet file!");
    Ok(())
}

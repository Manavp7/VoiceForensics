// Typed client for the VoiceForensics API.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type Verdict =
  | "AUTHENTIC"
  | "LEANING_AUTHENTIC"
  | "UNCERTAIN"
  | "LIKELY_SYNTHETIC"
  | "HIGH_CONFIDENCE_SYNTHETIC";

export interface Segment {
  start_ms: number;
  end_ms: number;
  score: number;
}

export interface Fingerprint {
  probable_source: string;
  confidence: number;
  alternative_sources: string[];
  distribution: Record<string, number>;
}

export interface DetectionResult {
  deepfake_probability: number;
  confidence_interval: [number, number];
  verdict: Verdict;
  uncertainty: number;
  naturalness_score: number | null;
  segments: Segment[];
  fingerprint: Fingerprint | null;
  metadata: {
    file_hash_sha256: string;
    duration_seconds: number | null;
    format: string | null;
    codec: string | null;
    edit_indicators: { edited: boolean; reasons: string[] };
  };
  quality: Record<string, number>;
}

export interface AnalysisResult {
  analysis_id: string;
  status: string;
  analysis_type: string;
  processing_time_ms: number;
  result: DetectionResult;
  provenance: {
    engine_version: string;
    active_detectors: string[];
    baseline_only: boolean;
    notes: string[];
  };
  report_url: string | null;
}

export async function analyzeUpload(
  file: File,
  analysisType: string,
): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("analysis_type", analysisType);
  const resp = await fetch(`${API_BASE}/v1/analyze/upload`, {
    method: "POST",
    body: form,
  });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail ?? `Request failed (${resp.status})`);
  }
  return resp.json();
}

export async function visualizeBlobUrl(
  file: File,
  kind: "mel" | "heatmap" | "waveform",
): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  form.append("kind", kind);
  const resp = await fetch(`${API_BASE}/v1/visualize/upload`, {
    method: "POST",
    body: form,
  });
  if (!resp.ok) throw new Error(`Visualization failed (${resp.status})`);
  const blob = await resp.blob();
  return URL.createObjectURL(blob);
}

export function reportUrl(analysisId: string): string {
  return `${API_BASE}/v1/reports/${analysisId}.pdf`;
}

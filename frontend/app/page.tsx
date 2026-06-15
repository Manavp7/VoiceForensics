"use client";

import { useState } from "react";
import {
  analyzeUpload,
  reportUrl,
  visualizeBlobUrl,
  type AnalysisResult,
} from "@/lib/api";
import { VerdictBadge } from "@/components/VerdictBadge";
import { ProbabilityGauge } from "@/components/ProbabilityGauge";
import { SegmentTimeline } from "@/components/SegmentTimeline";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [analysisType, setAnalysisType] = useState("full");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [images, setImages] = useState<Record<string, string>>({});

  async function onAnalyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setImages({});
    try {
      const res = await analyzeUpload(file, analysisType);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  }

  async function showImage(kind: "mel" | "heatmap" | "waveform") {
    if (!file) return;
    try {
      const url = await visualizeBlobUrl(file, kind);
      setImages((prev) => ({ ...prev, [kind]: url }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Visualization failed");
    }
  }

  const r = result?.result;

  return (
    <main className="mx-auto max-w-3xl px-6 py-10">
      <header className="mb-8">
        <h1 className="text-3xl font-bold">VoiceForensics</h1>
        <p className="text-gray-400">
          Forensic-grade audio deepfake detection
        </p>
      </header>

      <section className="rounded-lg border border-gray-800 bg-panel p-6">
        <label className="mb-3 block text-sm text-gray-400">Audio file</label>
        <input
          type="file"
          accept="audio/*,.wav,.mp3,.ogg,.m4a,.flac"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="block w-full text-sm text-gray-300 file:mr-4 file:rounded file:border-0 file:bg-accent file:px-4 file:py-2 file:text-white"
        />
        <div className="mt-4 flex items-center gap-3">
          <select
            value={analysisType}
            onChange={(e) => setAnalysisType(e.target.value)}
            className="rounded border border-gray-700 bg-ink px-3 py-2 text-sm"
          >
            <option value="quick">quick</option>
            <option value="full">full</option>
            <option value="legal">legal (PDF report)</option>
          </select>
          <button
            onClick={onAnalyze}
            disabled={!file || loading}
            className="rounded bg-accent px-5 py-2 text-sm font-semibold text-white disabled:opacity-40"
          >
            {loading ? "Analyzing…" : "Analyze"}
          </button>
        </div>
      </section>

      {error && (
        <div className="mt-6 rounded border border-red-700 bg-red-950 p-4 text-red-200">
          {error}
        </div>
      )}

      {r && result && (
        <section className="mt-8 space-y-6 rounded-lg border border-gray-800 bg-panel p-6">
          <div className="flex items-center justify-between">
            <VerdictBadge verdict={r.verdict} />
            <span className="text-xs text-gray-500">
              {result.analysis_id} · {result.processing_time_ms} ms
            </span>
          </div>

          {result.provenance.baseline_only && (
            <div className="rounded border border-yellow-700 bg-yellow-950/60 p-3 text-xs text-yellow-200">
              Heuristic baseline only — no trained neural weights configured.
              Scores are explainable signal-derived indicators.
            </div>
          )}

          <ProbabilityGauge
            probability={r.deepfake_probability}
            interval={r.confidence_interval}
          />

          <div className="grid grid-cols-2 gap-4 text-sm">
            <Stat label="Uncertainty" value={r.uncertainty.toFixed(3)} />
            <Stat
              label="Naturalness"
              value={r.naturalness_score?.toFixed(3) ?? "n/a"}
            />
            <Stat
              label="Probable source"
              value={r.fingerprint?.probable_source ?? "n/a"}
            />
            <Stat
              label="Duration"
              value={
                r.metadata.duration_seconds
                  ? `${r.metadata.duration_seconds.toFixed(1)}s`
                  : "n/a"
              }
            />
          </div>

          {r.segments.length > 0 && <SegmentTimeline segments={r.segments} />}

          <div>
            <div className="mb-1 text-sm text-gray-400">SHA-256</div>
            <code className="block break-all rounded bg-ink p-2 text-xs text-gray-300">
              {r.metadata.file_hash_sha256}
            </code>
          </div>

          <div className="flex flex-wrap gap-2">
            {(["mel", "heatmap", "waveform"] as const).map((k) => (
              <button
                key={k}
                onClick={() => showImage(k)}
                className="rounded border border-gray-700 px-3 py-1 text-xs hover:bg-gray-800"
              >
                Show {k}
              </button>
            ))}
            {result.report_url && (
              <a
                href={reportUrl(result.analysis_id)}
                target="_blank"
                rel="noreferrer"
                className="rounded bg-green-700 px-3 py-1 text-xs font-semibold text-white"
              >
                Download PDF report
              </a>
            )}
          </div>

          {Object.entries(images).map(([kind, url]) => (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              key={kind}
              src={url}
              alt={`${kind} exhibit`}
              className="w-full rounded border border-gray-800"
            />
          ))}
        </section>
      )}
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-gray-800 bg-ink p-3">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="font-mono">{value}</div>
    </div>
  );
}

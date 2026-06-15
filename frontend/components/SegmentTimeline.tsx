import type { Segment } from "@/lib/api";

export function SegmentTimeline({ segments }: { segments: Segment[] }) {
  if (!segments.length) return null;
  const total = segments[segments.length - 1].end_ms || 1;
  return (
    <div>
      <div className="mb-1 text-sm text-gray-400">Per-segment suspicion</div>
      <div className="flex h-12 w-full items-end gap-[1px] rounded bg-gray-900 p-1">
        {segments.map((s, i) => {
          const color =
            s.score < 0.4 ? "#1a7f37" : s.score < 0.6 ? "#9a6700" : "#cf222e";
          return (
            <div
              key={i}
              title={`${(s.start_ms / 1000).toFixed(1)}–${(s.end_ms / 1000).toFixed(1)}s : ${s.score.toFixed(2)}`}
              className="flex-1 rounded-sm"
              style={{ height: `${Math.max(s.score * 100, 4)}%`, background: color }}
            />
          );
        })}
      </div>
      <div className="mt-1 text-xs text-gray-500">
        0s – {(total / 1000).toFixed(1)}s
      </div>
    </div>
  );
}

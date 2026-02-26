export interface SectorData {
  id: string;
  name: string;
  stress_score: number;
  linguistic_mirroring: number;
  news_delay_predicted: number;
  complexity_index: number;
  temporal_ratio: { past: number; present: number; future: number };
  identity_gap: number;
  posts_per_minute: number;
  dehumanization_score: number;
  absolutist_triggers: number;
}

export interface TimeSeriesPoint {
  timestamp: string;
  complexity: number;
  temporal: number;
  identity_gap: number;
  blink_rate: number;
}

export interface NewsEvent {
  id: string;
  predicted_at: string;
  actual_at: string | null;
  headline: string;
  sector: string;
  delay_hours: number;
}

export interface EvidenceItem {
  id: string;
  type: "quote" | "pattern" | "metric";
  content: string;
  source: string;
  timestamp: string;
  severity: "low" | "medium" | "high" | "critical";
}

export interface Report {
  id: string;
  title: string;
  sector: string;
  confidence_score: number;
  generated_at: string;
  t_minus_hours: number;
  evidence: EvidenceItem[];
  summary: string;
}

export interface MockData {
  sectors: SectorData[];
  timeseries: TimeSeriesPoint[];
  news_events: NewsEvent[];
  reports: Report[];
}

import type { MediaItem } from "@/types";
import { MediaEmbed } from "./MediaEmbed";

interface MediaGalleryProps {
  items: MediaItem[];
  columns?: 1 | 2 | 3;
  className?: string;
}

const gridCols = {
  1: "grid-cols-1",
  2: "grid-cols-1 md:grid-cols-2",
  3: "grid-cols-1 md:grid-cols-2 lg:grid-cols-3",
} as const;

export function MediaGallery({
  items,
  columns = 2,
  className = "",
}: MediaGalleryProps) {
  if (items.length === 0) return null;

  return (
    <div className={`grid ${gridCols[columns]} gap-6 ${className}`}>
      {items.map((item, i) => (
        <MediaEmbed key={`${item.src}-${i}`} item={item} />
      ))}
    </div>
  );
}

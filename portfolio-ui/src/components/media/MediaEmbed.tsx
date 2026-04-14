import type { MediaItem } from "@/types";

interface MediaEmbedProps {
  item: MediaItem;
  className?: string;
}

/**
 * Universal media renderer — handles gifs, videos, and screenshots.
 * Drop media files in public/media/{gifs,videos,screenshots}/ and
 * reference them in content data files.
 */
export function MediaEmbed({ item, className = "" }: MediaEmbedProps) {
  const baseClass = `rounded-lg overflow-hidden ${className}`;

  switch (item.type) {
    case "video":
      return (
        <figure className={baseClass}>
          <video
            src={item.src}
            poster={item.poster}
            controls
            preload="metadata"
            className="w-full"
          >
            <track kind="captions" />
          </video>
          {item.caption && (
            <figcaption className="mt-2 text-sm text-surface-200/60 text-center">
              {item.caption}
            </figcaption>
          )}
        </figure>
      );

    case "gif":
      return (
        <figure className={baseClass}>
          <img
            src={item.src}
            alt={item.alt}
            loading="lazy"
            className="w-full"
          />
          {item.caption && (
            <figcaption className="mt-2 text-sm text-surface-200/60 text-center">
              {item.caption}
            </figcaption>
          )}
        </figure>
      );

    case "screenshot":
      return (
        <figure className={baseClass}>
          <div className="terminal-window">
            <div className="terminal-header">
              <span className="terminal-dot bg-red-500/80" />
              <span className="terminal-dot bg-yellow-500/80" />
              <span className="terminal-dot bg-green-500/80" />
              {item.caption && (
                <span className="ml-2 text-xs text-surface-200/50 font-mono">
                  {item.caption}
                </span>
              )}
            </div>
            <img
              src={item.src}
              alt={item.alt}
              loading="lazy"
              className="w-full"
            />
          </div>
        </figure>
      );
  }
}

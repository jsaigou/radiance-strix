// The mark: an R in three solid pieces, its stem, its bowl and its leg. It takes the current text
// colour.
export function Logo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 64 64" className={className} role="img" aria-label="radiance" fill="currentColor">
      <rect x="17" y="11" width="8" height="42" />
      <path d="M29 11h7a12.5 12.5 0 0 1 0 25h-7v-8h7a4.5 4.5 0 0 0 0-9h-7z" />
      <path d="M33 40h9l8 13h-9z" />
    </svg>
  )
}

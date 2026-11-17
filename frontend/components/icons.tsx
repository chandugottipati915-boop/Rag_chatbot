// A handful of inline icons, so the app needs no icon library.
type P = { className?: string };
const base = "h-4 w-4 shrink-0";

export const PlusIcon = ({ className = base }: P) => (
  <svg viewBox="0 0 16 16" className={className} fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
    <path d="M8 3v10M3 8h10" strokeLinecap="round" />
  </svg>
);
export const CloseIcon = ({ className = base }: P) => (
  <svg viewBox="0 0 16 16" className={className} fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
    <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
  </svg>
);
export const MenuIcon = ({ className = base }: P) => (
  <svg viewBox="0 0 16 16" className={className} fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
    <path d="M2.5 4.5h11M2.5 8h11M2.5 11.5h11" strokeLinecap="round" />
  </svg>
);
export const SendIcon = ({ className = base }: P) => (
  <svg viewBox="0 0 16 16" className={className} fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
    <path d="M8 13V3M3.5 7.5L8 3l4.5 4.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);
export const StopIcon = ({ className = base }: P) => (
  <svg viewBox="0 0 16 16" className={className} aria-hidden>
    <rect x="4" y="4" width="8" height="8" rx="1.5" fill="currentColor" />
  </svg>
);

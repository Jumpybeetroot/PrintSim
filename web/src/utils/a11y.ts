import type React from "react";

/**
 * Keyboard support for a `role="tablist"`: Left/Right move to the neighbouring tab,
 * Home/End to the first/last, and the tab is activated as it receives focus.
 */
export function onTabListKeyDown(e: React.KeyboardEvent<HTMLElement>): void {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
  const tabs = Array.from(e.currentTarget.querySelectorAll<HTMLElement>('[role="tab"]'));
  const current = tabs.indexOf(document.activeElement as HTMLElement);
  if (current < 0) return;
  e.preventDefault();
  const next =
    e.key === "Home"
      ? 0
      : e.key === "End"
        ? tabs.length - 1
        : (current + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
  tabs[next].focus();
  tabs[next].click();
}

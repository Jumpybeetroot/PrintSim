import React, { useEffect, useLayoutEffect, useRef, useState } from "react";

/**
 * One tooltip for the whole app, driven by the `title` attributes that already carry
 * the explanations. Native `title` tooltips only appear on mouse hover; this layer
 * also shows them on keyboard focus and on tap (for non-interactive elements), lets
 * Escape dismiss them, keeps them open while the pointer is over the tooltip, and
 * follows the anchor when the page scrolls.
 *
 * While a tooltip is showing, the element's `title` is parked in `data-tip` so the
 * browser's own tooltip does not appear on top, and restored when it hides.
 */

const TOOLTIP_ID = "studio-tooltip";
const HOVER_DELAY_MS = 350;
const HIDE_DELAY_MS = 120;
const INTERACTIVE = "button, a[href], input, select, textarea, summary, [role='button'], [role='tab']";

interface TipState {
  text: string;
  rect: DOMRect;
}

/** The element whose explanation applies to `target`, if any. */
function findTipSource(target: Element | null): HTMLElement | null {
  if (!target) return null;
  const own = target.closest<HTMLElement>("[title]:not([title='']), [data-tip]");
  if (own) return own;
  // A focused control with no title of its own: use its label's explanation.
  if (target instanceof HTMLElement && target.id) {
    return document.querySelector<HTMLElement>(
      `label[for="${CSS.escape(target.id)}"][title]:not([title=''])`
    );
  }
  return null;
}

export const TooltipLayer: React.FC = () => {
  const [tip, setTip] = useState<TipState | null>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);
  const sourceRef = useRef<HTMLElement | null>(null);
  const anchorRef = useRef<Element | null>(null);
  const showTimer = useRef<number | null>(null);
  const hideTimer = useRef<number | null>(null);

  useEffect(() => {
    const clearTimers = () => {
      if (showTimer.current !== null) window.clearTimeout(showTimer.current);
      if (hideTimer.current !== null) window.clearTimeout(hideTimer.current);
      showTimer.current = null;
      hideTimer.current = null;
    };

    const restoreTitle = () => {
      const src = sourceRef.current;
      if (src && src.dataset.tip !== undefined) {
        if (!src.hasAttribute("title")) src.setAttribute("title", src.dataset.tip);
        delete src.dataset.tip;
      }
      const anchor = anchorRef.current;
      if (anchor?.getAttribute("aria-describedby") === TOOLTIP_ID) {
        anchor.removeAttribute("aria-describedby");
      }
      sourceRef.current = null;
      anchorRef.current = null;
    };

    const hide = () => {
      clearTimers();
      restoreTitle();
      setTip(null);
      setPos(null);
    };

    const show = (source: HTMLElement, anchor: Element) => {
      clearTimers();
      if (sourceRef.current === source && anchorRef.current === anchor) return;
      restoreTitle();
      const text = source.getAttribute("title") || source.dataset.tip || "";
      if (!text.trim()) return;
      source.dataset.tip = text;
      source.removeAttribute("title");
      sourceRef.current = source;
      anchorRef.current = anchor;
      anchor.setAttribute("aria-describedby", TOOLTIP_ID);
      setPos(null);
      setTip({ text, rect: anchor.getBoundingClientRect() });
    };

    // Scrolling (including the scroll that brings a newly focused control into view)
    // moves the anchor: follow it, and hide only once it has left the window.
    const follow = () => {
      const anchor = anchorRef.current;
      if (!anchor) return;
      const rect = anchor.getBoundingClientRect();
      if (rect.bottom < 0 || rect.top > window.innerHeight || rect.width === 0) {
        hide();
        return;
      }
      setTip((prev) => (prev ? { ...prev, rect } : prev));
    };

    const scheduleHide = () => {
      if (showTimer.current !== null) window.clearTimeout(showTimer.current);
      showTimer.current = null;
      if (hideTimer.current === null) hideTimer.current = window.setTimeout(hide, HIDE_DELAY_MS);
    };

    const onMouseOver = (e: MouseEvent) => {
      const target = e.target as Element | null;
      if (target && boxRef.current?.contains(target)) {
        // Pointer moved onto the tooltip itself: keep it open.
        if (hideTimer.current !== null) window.clearTimeout(hideTimer.current);
        hideTimer.current = null;
        return;
      }
      const source = findTipSource(target);
      if (!source) {
        if (sourceRef.current) scheduleHide();
        return;
      }
      if (source === sourceRef.current) {
        if (hideTimer.current !== null) window.clearTimeout(hideTimer.current);
        hideTimer.current = null;
        return;
      }
      if (showTimer.current !== null) window.clearTimeout(showTimer.current);
      showTimer.current = window.setTimeout(() => show(source, source), HOVER_DELAY_MS);
    };

    const onMouseOut = (e: MouseEvent) => {
      const to = e.relatedTarget as Element | null;
      if (to && (sourceRef.current?.contains(to) || boxRef.current?.contains(to))) return;
      scheduleHide();
    };

    const onFocusIn = (e: FocusEvent) => {
      const target = e.target as Element | null;
      // Keyboard focus only: a mouse click should not pop a tooltip over what was clicked.
      if (!target || !target.matches(":focus-visible")) return;
      const source = findTipSource(target);
      if (source) show(source, target);
    };

    const onFocusOut = () => {
      if (sourceRef.current) hide();
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && sourceRef.current) hide();
    };

    // Touch: tapping an explanatory (non-interactive) element shows its tooltip;
    // tapping anywhere else dismisses it. Buttons and inputs keep their normal tap.
    const onPointerDown = (e: PointerEvent) => {
      if (e.pointerType !== "touch") {
        if (sourceRef.current) hide();
        return;
      }
      const target = e.target as Element | null;
      if (target && boxRef.current?.contains(target)) return;
      const source = findTipSource(target);
      if (source && source !== sourceRef.current && !target?.closest(INTERACTIVE)) {
        show(source, source);
      } else if (sourceRef.current) {
        hide();
      }
    };

    document.addEventListener("mouseover", onMouseOver);
    document.addEventListener("mouseout", onMouseOut);
    document.addEventListener("focusin", onFocusIn);
    document.addEventListener("focusout", onFocusOut);
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("scroll", follow, true);
    window.addEventListener("resize", hide);
    return () => {
      document.removeEventListener("mouseover", onMouseOver);
      document.removeEventListener("mouseout", onMouseOut);
      document.removeEventListener("focusin", onFocusIn);
      document.removeEventListener("focusout", onFocusOut);
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("scroll", follow, true);
      window.removeEventListener("resize", hide);
      clearTimers();
      restoreTitle();
    };
  }, []);

  // Place below the anchor, or above when there is no room; clamp to the window.
  useLayoutEffect(() => {
    const box = boxRef.current;
    if (!tip || !box) return;
    const margin = 8;
    const { width, height } = box.getBoundingClientRect();
    const { rect } = tip;
    let top = rect.bottom + 6;
    if (top + height > window.innerHeight - margin) top = Math.max(margin, rect.top - height - 6);
    const centred = rect.left + rect.width / 2 - width / 2;
    const left = Math.min(Math.max(margin, centred), Math.max(margin, window.innerWidth - width - margin));
    setPos({ left, top });
  }, [tip]);

  return (
    <div
      id={TOOLTIP_ID}
      ref={boxRef}
      role="tooltip"
      className="studio-tooltip"
      hidden={!tip}
      style={{
        left: pos?.left ?? 0,
        top: pos?.top ?? 0,
        visibility: tip && pos ? "visible" : "hidden",
      }}
    >
      {tip?.text}
    </div>
  );
};

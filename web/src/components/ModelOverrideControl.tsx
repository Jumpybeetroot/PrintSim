import React, { useEffect, useRef, useState } from "react";
import {
  ModelOverrideGroupKey,
  ModelOverridesData,
  PrinterSpecData,
  SimulationResponse,
} from "../types";

export interface ModelOverrideFieldDef {
  key: string;
  label: string;
  unit?: string;
  type?: "number" | "boolean";
  min?: number;
  max?: number;
  step?: number;
  decimals?: number;
  fallbackBaseline: number | boolean;
  tooltip?: string;
}

export interface ModelOverrideControlProps {
  visible: boolean;
  idPrefix: string;
  title: string;
  group: ModelOverrideGroupKey;
  fields: ModelOverrideFieldDef[];
  spec: PrinterSpecData;
  sim: SimulationResponse | null;
  update: (mutator: (draft: PrinterSpecData) => void, changeLabel?: string) => void;
  note?: string;
}

export interface PresetAffectsItem {
  label: string;
  value: string;
}

export interface PresetAffectsStripProps {
  visible: boolean;
  items: PresetAffectsItem[];
  suffix?: string;
}

type OverrideGroupRecord = Record<string, number | boolean | null | undefined>;

function getGroupOverrides(
  overrides: ModelOverridesData | null | undefined,
  group: ModelOverrideGroupKey
): OverrideGroupRecord | null {
  if (!overrides) return null;
  const raw = overrides[group];
  if (!raw || typeof raw !== "object") return null;
  return raw as OverrideGroupRecord;
}

export function countActiveModelOverrides(overrides?: ModelOverridesData | null): number {
  if (!overrides) return 0;
  let count = 0;
  for (const groupVal of Object.values(overrides)) {
    if (!groupVal || typeof groupVal !== "object") continue;
    for (const fieldVal of Object.values(groupVal as OverrideGroupRecord)) {
      if (fieldVal !== null && fieldVal !== undefined) {
        count += 1;
      }
    }
  }
  return count;
}

export function clearModelOverrideGroup(
  draft: PrinterSpecData,
  group: ModelOverrideGroupKey
): boolean {
  const rec = getGroupOverrides(draft.model_overrides, group);
  if (!rec) return false;
  const hadAny = Object.values(rec).some((v) => v !== null && v !== undefined);
  if (draft.model_overrides) {
    delete draft.model_overrides[group];
    const hasOtherGroups = Object.values(draft.model_overrides).some(
      (g) => g && typeof g === "object" && Object.values(g as OverrideGroupRecord).some((v) => v !== null && v !== undefined)
    );
    if (!hasOtherGroups) {
      draft.model_overrides = null;
    }
  }
  return hadAny;
}

export function clearAllModelOverrides(draft: PrinterSpecData): boolean {
  const count = countActiveModelOverrides(draft.model_overrides);
  draft.model_overrides = null;
  return count > 0;
}

function setFieldOverrideInDraft(
  draft: PrinterSpecData,
  group: ModelOverrideGroupKey,
  fieldKey: string,
  value: number | boolean | null
): void {
  if (value === null) {
    const rec = getGroupOverrides(draft.model_overrides, group);
    if (!rec) return;
    delete rec[fieldKey];
    const remainingInGroup = Object.values(rec).some((v) => v !== null && v !== undefined);
    if (!remainingInGroup && draft.model_overrides) {
      delete draft.model_overrides[group];
    }
    const remainingTotal = countActiveModelOverrides(draft.model_overrides);
    if (remainingTotal === 0) {
      draft.model_overrides = null;
    }
    return;
  }

  if (!draft.model_overrides) {
    draft.model_overrides = {};
  }
  const existing = getGroupOverrides(draft.model_overrides, group);
  const nextGroup: OverrideGroupRecord = existing ? { ...existing } : {};
  nextGroup[fieldKey] = value;
  (draft.model_overrides as Record<string, OverrideGroupRecord>)[group] = nextGroup;
}

function formatInputNumber(val: number, field: ModelOverrideFieldDef): string {
  if (!Number.isFinite(val)) return "";
  if (field.decimals != null) {
    return String(Number(val.toFixed(field.decimals)));
  }
  return String(Number(val.toPrecision(6)));
}

function formatReadoutValue(val: number | boolean, field: ModelOverrideFieldDef): string {
  if (field.type === "boolean" || typeof val === "boolean") {
    return val ? "Yes" : "No";
  }
  const num = Number(val);
  if (!Number.isFinite(num)) return "—";
  const decimals = field.decimals ?? (Math.abs(num) >= 100 ? 0 : 2);
  let numStr: string;
  if (decimals === 0) {
    numStr = Math.round(num).toLocaleString();
  } else {
    numStr = num.toFixed(decimals);
  }
  if (!field.unit) return numStr;
  if (field.unit === "×" || field.unit === "×C") {
    return `${numStr}${field.unit}`;
  }
  return `${numStr} ${field.unit}`;
}

export const ModelOverrideControl: React.FC<ModelOverrideControlProps> = ({
  visible,
  idPrefix,
  title,
  group,
  fields,
  spec,
  sim,
  update,
  note,
}) => {
  const effGroup = sim?.effective_model_params?.[group];
  const specGroup = getGroupOverrides(spec.model_overrides, group);

  const resolvedFields = fields.map((field) => {
    const rawBaseline = effGroup?.baseline?.[field.key];
    const baseline = rawBaseline !== undefined ? rawBaseline : field.fallbackBaseline;
    const rawOverride = specGroup?.[field.key];
    const isOverridden = rawOverride !== null && rawOverride !== undefined;
    const rawEffective = effGroup?.effective?.[field.key];
    const effective = isOverridden
      ? (rawOverride as number | boolean)
      : rawEffective !== undefined
        ? rawEffective
        : baseline;
    return { field, baseline, effective, isOverridden };
  });

  const hasAnyOverride = resolvedFields.some((rf) => rf.isOverridden);

  const [expanded, setExpanded] = useState<boolean>(hasAnyOverride);
  const prevHasOverrideRef = useRef<boolean>(hasAnyOverride);
  const focusedKeyRef = useRef<string | null>(null);
  const timersRef = useRef<Record<string, number>>({});

  const [drafts, setDrafts] = useState<Record<string, string>>(() => {
    const init: Record<string, string> = {};
    for (const rf of resolvedFields) {
      if (rf.field.type !== "boolean") {
        init[rf.field.key] = formatInputNumber(Number(rf.effective), rf.field);
      }
    }
    return init;
  });

  useEffect(() => {
    if (hasAnyOverride && !prevHasOverrideRef.current) {
      setExpanded(true);
    }
    prevHasOverrideRef.current = hasAnyOverride;
  }, [hasAnyOverride]);

  // Sync external effective value updates into local input drafts when not editing that input.
  const effectiveSignature = resolvedFields
    .map((rf) => `${rf.field.key}:${String(rf.effective)}`)
    .join("|");

  useEffect(() => {
    setDrafts((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const rf of resolvedFields) {
        if (rf.field.type === "boolean") continue;
        if (focusedKeyRef.current === rf.field.key) continue;
        const formatted = formatInputNumber(Number(rf.effective), rf.field);
        if (next[rf.field.key] !== formatted) {
          next[rf.field.key] = formatted;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveSignature]);

  useEffect(() => {
    return () => {
      for (const timerId of Object.values(timersRef.current)) {
        window.clearTimeout(timerId);
      }
      timersRef.current = {};
    };
  }, []);

  if (!visible) return null;

  const clearTimer = (fieldKey: string) => {
    const existing = timersRef.current[fieldKey];
    if (existing != null) {
      window.clearTimeout(existing);
      delete timersRef.current[fieldKey];
    }
  };

  const commitNumberField = (
    field: ModelOverrideFieldDef,
    baselineVal: number,
    effectiveVal: number,
    wasOverridden: boolean,
    rawStr: string
  ) => {
    clearTimer(field.key);
    const trimmed = rawStr.trim();

    if (trimmed === "") {
      setDrafts((prev) => ({
        ...prev,
        [field.key]: formatInputNumber(baselineVal, field),
      }));
      if (wasOverridden) {
        update(
          (d) => {
            setFieldOverrideInDraft(d, group, field.key, null);
          },
          `${title} · ${field.label}: Reset to default (${formatReadoutValue(baselineVal, field)})`
        );
      }
      return;
    }

    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed)) {
      setDrafts((prev) => ({
        ...prev,
        [field.key]: formatInputNumber(effectiveVal, field),
      }));
      return;
    }

    let clamped = parsed;
    if (field.min != null) clamped = Math.max(field.min, clamped);
    if (field.max != null) clamped = Math.min(field.max, clamped);

    const tol = Math.max(1e-6, Math.abs(baselineVal) * 1e-5);
    const matchesBaseline = Math.abs(clamped - baselineVal) <= tol;

    if (matchesBaseline) {
      setDrafts((prev) => ({
        ...prev,
        [field.key]: formatInputNumber(baselineVal, field),
      }));
      if (wasOverridden) {
        update(
          (d) => {
            setFieldOverrideInDraft(d, group, field.key, null);
          },
          `${title} · ${field.label}: Reset to default (${formatReadoutValue(baselineVal, field)})`
        );
      }
      return;
    }

    const effTol = Math.max(1e-6, Math.abs(effectiveVal) * 1e-5);
    if (Math.abs(clamped - effectiveVal) <= effTol && wasOverridden) {
      setDrafts((prev) => ({
        ...prev,
        [field.key]: formatInputNumber(clamped, field),
      }));
      return;
    }

    setDrafts((prev) => ({
      ...prev,
      [field.key]: formatInputNumber(clamped, field),
    }));

    update(
      (d) => {
        setFieldOverrideInDraft(d, group, field.key, clamped);
      },
      `${title} · ${field.label}: ${formatReadoutValue(effectiveVal, field)} → ${formatReadoutValue(clamped, field)} (override)`
    );
  };

  const handleNumberChange = (
    field: ModelOverrideFieldDef,
    baselineVal: number,
    effectiveVal: number,
    wasOverridden: boolean,
    rawStr: string
  ) => {
    setDrafts((prev) => ({ ...prev, [field.key]: rawStr }));
    clearTimer(field.key);
    const trimmed = rawStr.trim();
    if (trimmed === "" || trimmed === "-" || trimmed.endsWith(".")) {
      return;
    }
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed)) return;
    if (field.min != null && parsed < field.min) return;
    if (field.max != null && parsed > field.max) return;
    timersRef.current[field.key] = window.setTimeout(() => {
      commitNumberField(field, baselineVal, effectiveVal, wasOverridden, rawStr);
    }, 250);
  };

  const commitBooleanField = (
    field: ModelOverrideFieldDef,
    baselineVal: boolean,
    effectiveVal: boolean,
    nextVal: boolean
  ) => {
    if (nextVal === effectiveVal) return;
    const matchesBaseline = nextVal === baselineVal;
    update(
      (d) => {
        setFieldOverrideInDraft(d, group, field.key, matchesBaseline ? null : nextVal);
      },
      matchesBaseline
        ? `${title} · ${field.label}: Reset to default (${formatReadoutValue(baselineVal, field)})`
        : `${title} · ${field.label}: ${formatReadoutValue(effectiveVal, field)} → ${formatReadoutValue(nextVal, field)} (override)`
    );
  };

  const handleResetGroup = () => {
    for (const field of fields) {
      clearTimer(field.key);
    }
    update(
      (d) => {
        clearModelOverrideGroup(d, group);
      },
      `${title}: Reset overrides to catalog default`
    );
  };

  return (
    <div className={`model-override-strip${hasAnyOverride ? " overridden" : ""}`}>
      <div className="model-override-head">
        <div className="model-override-summary">
          <span className="model-override-prefix">Sets:</span>
          {resolvedFields.map(({ field, baseline, effective, isOverridden }, idx) => (
            <React.Fragment key={field.key}>
              {idx > 0 && (
                <span className="model-override-sep" aria-hidden="true">
                  ·
                </span>
              )}
              <span
                className={`model-override-item${isOverridden ? " is-overridden" : ""}`}
                title={
                  field.tooltip
                    ? `${field.tooltip} (Catalog default: ${formatReadoutValue(baseline, field)})`
                    : `Catalog default: ${formatReadoutValue(baseline, field)}`
                }
              >
                <span className="model-override-item-lbl">{field.label}</span>{" "}
                <strong className="model-override-item-val">
                  {formatReadoutValue(effective, field)}
                </strong>
              </span>
            </React.Fragment>
          ))}
        </div>
        <div className="model-override-actions">
          {hasAnyOverride && <span className="badge-pill info">Overridden</span>}
          {hasAnyOverride && (
            <button
              type="button"
              className="override-action-btn"
              onClick={handleResetGroup}
              aria-label={`Reset ${title} overrides to catalog default`}
              title={`Reset ${title} overrides to catalog default`}
            >
              Reset
            </button>
          )}
          <button
            type="button"
            className={`override-action-btn${expanded ? " active" : ""}`}
            aria-pressed={expanded}
            aria-label={`${expanded ? "Hide" : "Override"} ${title} physics parameters`}
            onClick={() => setExpanded((prev) => !prev)}
          >
            {expanded ? "Done" : "Override"}
          </button>
        </div>
      </div>

      {note && <div className="model-override-note">{note}</div>}

      {expanded && (
        <div className={`model-override-grid${fields.length === 1 ? " single" : ""}`}>
          {resolvedFields.map(({ field, baseline, effective, isOverridden }) => {
            const inputId = `${idPrefix}-${field.key}`;
            const defaultLabel = formatReadoutValue(baseline, field);
            const fieldTitle = field.tooltip
              ? `${field.tooltip} — Catalog default: ${defaultLabel}`
              : `Catalog default: ${defaultLabel}`;

            if (field.type === "boolean") {
              return (
                <div key={field.key} className="model-override-field model-override-field-bool">
                  <label
                    className={`checkbox-row${isOverridden ? " is-overridden" : ""}`}
                    title={fieldTitle}
                  >
                    <input
                      id={inputId}
                      type="checkbox"
                      checked={Boolean(effective)}
                      onChange={(e) =>
                        commitBooleanField(
                          field,
                          Boolean(baseline),
                          Boolean(effective),
                          e.target.checked
                        )
                      }
                    />
                    <span>{field.label}</span>
                  </label>
                </div>
              );
            }

            const baseNum = Number(baseline);
            const effNum = Number(effective);
            const draftVal = drafts[field.key] ?? formatInputNumber(effNum, field);
            const labelText = field.unit ? `${field.label} (${field.unit})` : field.label;

            return (
              <div key={field.key} className="model-override-field">
                <label
                  htmlFor={inputId}
                  className={isOverridden ? "is-overridden" : undefined}
                  title={fieldTitle}
                >
                  <span>{labelText}</span>
                </label>
                <input
                  id={inputId}
                  type="number"
                  min={field.min}
                  max={field.max}
                  step={field.step ?? "any"}
                  value={draftVal}
                  placeholder={formatInputNumber(baseNum, field)}
                  title={fieldTitle}
                  onFocus={() => {
                    focusedKeyRef.current = field.key;
                  }}
                  onChange={(e) =>
                    handleNumberChange(field, baseNum, effNum, isOverridden, e.target.value)
                  }
                  onBlur={() => {
                    focusedKeyRef.current = null;
                    commitNumberField(field, baseNum, effNum, isOverridden, draftVal);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      commitNumberField(field, baseNum, effNum, isOverridden, draftVal);
                      e.currentTarget.blur();
                    } else if (e.key === "Escape") {
                      clearTimer(field.key);
                      setDrafts((prev) => ({
                        ...prev,
                        [field.key]: formatInputNumber(effNum, field),
                      }));
                      e.currentTarget.blur();
                    }
                  }}
                />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export const PresetAffectsStrip: React.FC<PresetAffectsStripProps> = ({
  visible,
  items,
  suffix,
}) => {
  if (!visible || items.length === 0) return null;

  return (
    <div className="model-override-strip preset-strip">
      <div className="model-override-summary">
        <span className="model-override-prefix">Preset updates:</span>
        {items.map((item, idx) => (
          <React.Fragment key={`${item.label}-${idx}`}>
            {idx > 0 && (
              <span className="model-override-sep" aria-hidden="true">
                ·
              </span>
            )}
            <span className="model-override-item">
              {item.label ? (
                <>
                  <span className="model-override-item-lbl">{item.label}</span>{" "}
                </>
              ) : null}
              <strong className="model-override-item-val">{item.value}</strong>
            </span>
          </React.Fragment>
        ))}
        {suffix && <span className="model-override-suffix">{suffix}</span>}
      </div>
    </div>
  );
};

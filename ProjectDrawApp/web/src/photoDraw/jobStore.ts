/**
 * In-memory + session Photo Draw job store.
 * Survives route changes; last success/pending restored when returning to /photo-draw.
 */

import { create } from "zustand";
import type { BoardRect, ValidatedGuide } from "@shared/photoDrawGuide";
import type { StepStrokeGuide } from "@shared/strokeGuides";

export type PhotoDrawJobStatus =
  | "idle"
  | "running"
  | "ok"
  | "fail"
  | "cancelled";

export type PhotoDrawJobSnapshot = {
  status: PhotoDrawJobStatus;
  jobId: string | null;
  fileName: string | null;
  /** Original or focused preview (blob: or data:) */
  previewUrl: string | null;
  progress: string | null;
  error: string | null;
  meta: string | null;
  focusNote: string | null;
  debugScores: string | null;
  steps: StepStrokeGuide[] | null;
  stepIndex: number;
  board: BoardRect | null;
  strokeCount: number | null;
  mode: string | null;
  attemptLabel: string | null;
  /** Banner / toast when job completes off-page */
  toast: { kind: "ok" | "fail"; message: string } | null;
  /** User left /photo-draw while running */
  notifyWhenDone: boolean;
};

type Actions = {
  startJob: (p: {
    jobId: string;
    fileName: string;
    previewUrl: string;
    notifyWhenDone?: boolean;
  }) => void;
  setProgress: (jobId: string, progress: string) => void;
  setNotifyWhenDone: (v: boolean) => void;
  completeOk: (
    jobId: string,
    p: {
      previewUrl: string;
      result: Extract<ValidatedGuide, { ok: true }>;
      focusNote: string | null;
      debugScores: string | null;
      meta: string;
    },
  ) => void;
  completeFail: (
    jobId: string,
    p: {
      previewUrl?: string;
      error: string;
      meta: string;
      debugScores: string | null;
      focusNote: string | null;
    },
  ) => void;
  cancelJob: (jobId?: string) => void;
  clearToast: () => void;
  setStepIndex: (i: number) => void;
  setShowBusyClear: () => void;
  hydrateFromSession: () => void;
  persistSuccessToSession: () => void;
};

// Bump when stroke pipeline semantics change so stale lessons (e.g. drawing
 // misclassified as photo → 76-stroke silhouette) are not restored.
const PIPELINE_VERSION = 7;
const SESSION_KEY = `qp-photo-draw-job-v${PIPELINE_VERSION}`;

const initial: PhotoDrawJobSnapshot = {
  status: "idle",
  jobId: null,
  fileName: null,
  previewUrl: null,
  progress: null,
  error: null,
  meta: null,
  focusNote: null,
  debugScores: null,
  steps: null,
  stepIndex: 0,
  board: null,
  strokeCount: null,
  mode: null,
  attemptLabel: null,
  toast: null,
  notifyWhenDone: true,
};

function persistLite(state: PhotoDrawJobSnapshot) {
  try {
    if (state.status !== "ok" || !state.steps) {
      sessionStorage.removeItem(SESSION_KEY);
      return;
    }
    const payload = {
      pipelineVersion: PIPELINE_VERSION,
      status: state.status,
      fileName: state.fileName,
      previewUrl:
        state.previewUrl && state.previewUrl.startsWith("data:")
          ? state.previewUrl
          : null,
      meta: state.meta,
      focusNote: state.focusNote,
      steps: state.steps,
      board: state.board,
      strokeCount: state.strokeCount,
      mode: state.mode,
      attemptLabel: state.attemptLabel,
    };
    sessionStorage.setItem(SESSION_KEY, JSON.stringify(payload));
    // Drop older session keys so a hard refresh cannot resurrect bad lessons.
    try {
      for (let v = 1; v < PIPELINE_VERSION; v++) {
        sessionStorage.removeItem(`qp-photo-draw-job-v${v}`);
      }
    } catch {
      /* */
    }
  } catch {
    /* quota */
  }
}

export const usePhotoDrawJob = create<PhotoDrawJobSnapshot & Actions>(
  (set, get) => ({
    ...initial,
    startJob: ({ jobId, fileName, previewUrl, notifyWhenDone }) =>
      set({
        status: "running",
        jobId,
        fileName,
        previewUrl,
        progress: "Starting…",
        error: null,
        meta: null,
        focusNote: null,
        debugScores: null,
        steps: null,
        stepIndex: 0,
        board: null,
        strokeCount: null,
        mode: null,
        attemptLabel: null,
        toast: null,
        notifyWhenDone: notifyWhenDone ?? true,
      }),
    setProgress: (jobId, progress) => {
      if (get().jobId !== jobId) return;
      set({ progress });
    },
    setNotifyWhenDone: (v) => set({ notifyWhenDone: v }),
    completeOk: (jobId, p) => {
      if (get().jobId !== jobId) return;
      const next: PhotoDrawJobSnapshot = {
        ...get(),
        status: "ok",
        jobId,
        previewUrl: p.previewUrl,
        progress: null,
        error: null,
        meta: p.meta,
        focusNote: p.focusNote,
        debugScores: p.debugScores,
        steps: p.result.steps,
        stepIndex: 0,
        board: p.result.board,
        strokeCount: p.result.strokeCount,
        mode: p.result.mode,
        attemptLabel: p.result.attemptLabel,
        toast: {
          kind: "ok",
          message: "Your stroke lesson is ready — open AI Tutor to follow along.",
        },
      };
      set(next);
      persistLite(next);
    },
    completeFail: (jobId, p) => {
      if (get().jobId !== jobId) return;
      set({
        status: "fail",
        jobId,
        previewUrl: p.previewUrl ?? get().previewUrl,
        progress: null,
        error: p.error,
        meta: p.meta,
        debugScores: p.debugScores,
        focusNote: p.focusNote,
        steps: null,
        board: null,
        toast: {
          kind: "fail",
          message:
            "That photo didn’t yield a clear lesson yet. Open AI Tutor to try again.",
        },
      });
      try {
        sessionStorage.removeItem(SESSION_KEY);
      } catch {
        /* */
      }
    },
    cancelJob: (jobId) => {
      if (jobId && get().jobId && get().jobId !== jobId) return;
      set({
        status: "cancelled",
        progress: null,
        toast: null,
      });
    },
    clearToast: () => set({ toast: null }),
    setStepIndex: (i) => set({ stepIndex: i }),
    setShowBusyClear: () => set({ progress: null }),
    hydrateFromSession: () => {
      try {
        // Always clear pre-v2 sessions (drawing→photo silhouette regressions).
        try {
          for (let v = 1; v < PIPELINE_VERSION; v++) {
            sessionStorage.removeItem(`qp-photo-draw-job-v${v}`);
          }
        } catch {
          /* */
        }
        const raw = sessionStorage.getItem(SESSION_KEY);
        if (!raw) return;
        const data = JSON.parse(raw) as Partial<PhotoDrawJobSnapshot> & {
          pipelineVersion?: number;
        };
        if (data.pipelineVersion !== PIPELINE_VERSION) {
          sessionStorage.removeItem(SESSION_KEY);
          return;
        }
        if (data.status === "ok" && data.steps?.length) {
          set({
            ...get(),
            status: "ok",
            fileName: data.fileName ?? null,
            previewUrl: data.previewUrl ?? get().previewUrl,
            meta: data.meta ?? null,
            focusNote: data.focusNote ?? null,
            steps: data.steps,
            board: data.board ?? null,
            strokeCount: data.strokeCount ?? null,
            mode: data.mode ?? null,
            attemptLabel: data.attemptLabel ?? null,
            error: null,
            progress: null,
            stepIndex: 0,
          });
        }
      } catch {
        /* */
      }
    },
    persistSuccessToSession: () => persistLite(get()),
  }),
);

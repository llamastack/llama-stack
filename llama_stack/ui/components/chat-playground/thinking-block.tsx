/**
 * ThinkingBlock Component
 * Displays LLM thinking/reasoning content in a collapsible, animated block
 * Shows duration, pulsing animation during streaming, and expandable content
 */

"use client";

import { useState } from "react";
import { ChevronDown, Brain } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";

export interface ThinkingPart {
  type: "thinking";
  content: string;
  startTime?: number;
  endTime?: number;
}

interface ThinkingBlockProps {
  part: ThinkingPart;
  isStreaming?: boolean;
}

/**
 * Formats duration in milliseconds to human-readable format
 */
function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`;
  }
  const seconds = (ms / 1000).toFixed(1);
  return `${seconds}s`;
}

/**
 * Calculates duration from start and end times
 */
function calculateDuration(
  startTime?: number,
  endTime?: number
): number | null {
  if (!startTime) return null;
  const end = endTime || Date.now();
  return end - startTime;
}

export function ThinkingBlock({
  part,
  isStreaming = false,
}: ThinkingBlockProps) {
  const [isOpen, setIsOpen] = useState(false);

  const duration = calculateDuration(part.startTime, part.endTime);
  const isComplete = !!part.endTime;
  const isPulsing = isStreaming || !isComplete;

  return (
    <Collapsible
      open={isOpen}
      onOpenChange={setIsOpen}
      className="my-2 rounded-lg border border-purple-200 bg-purple-50 dark:border-purple-800 dark:bg-purple-950/30"
    >
      <CollapsibleTrigger className="flex w-full items-center justify-between px-4 py-2.5 hover:bg-purple-100 dark:hover:bg-purple-900/30 transition-colors rounded-lg">
        <div className="flex items-center gap-2">
          {/* Pulsing Brain Icon */}
          <motion.div
            animate={
              isPulsing
                ? {
                    scale: [1, 1.1, 1],
                    opacity: [0.7, 1, 0.7],
                  }
                : { scale: 1, opacity: 0.8 }
            }
            transition={
              isPulsing
                ? {
                    duration: 1.5,
                    repeat: Infinity,
                    ease: "easeInOut",
                  }
                : {}
            }
          >
            <Brain className="h-4 w-4 text-purple-600 dark:text-purple-400" />
          </motion.div>

          {/* Label */}
          <span className="text-sm font-medium text-purple-700 dark:text-purple-300">
            {isPulsing ? "Thinking..." : "Thought Process"}
          </span>

          {/* Duration Badge */}
          {duration !== null && (
            <motion.span
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-xs font-mono bg-purple-200 dark:bg-purple-800 text-purple-800 dark:text-purple-200 px-2 py-0.5 rounded-full"
            >
              {formatDuration(duration)}
            </motion.span>
          )}

          {/* Streaming Indicator Dots */}
          {isPulsing && (
            <motion.div
              className="flex gap-1"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              {[0, 1, 2].map(i => (
                <motion.div
                  key={i}
                  className="w-1 h-1 bg-purple-500 dark:bg-purple-400 rounded-full"
                  animate={{
                    y: [0, -4, 0],
                  }}
                  transition={{
                    duration: 0.6,
                    repeat: Infinity,
                    delay: i * 0.15,
                    ease: "easeInOut",
                  }}
                />
              ))}
            </motion.div>
          )}
        </div>

        {/* Chevron Toggle Icon */}
        <motion.div
          animate={{ rotate: isOpen ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronDown className="h-4 w-4 text-purple-600 dark:text-purple-400" />
        </motion.div>
      </CollapsibleTrigger>

      <AnimatePresence initial={false}>
        {isOpen && (
          <CollapsibleContent forceMount>
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{
                height: "auto",
                opacity: 1,
                transition: {
                  height: {
                    duration: 0.3,
                    ease: "easeOut",
                  },
                  opacity: {
                    duration: 0.2,
                    delay: 0.1,
                  },
                },
              }}
              exit={{
                height: 0,
                opacity: 0,
                transition: {
                  height: {
                    duration: 0.3,
                    ease: "easeIn",
                  },
                  opacity: {
                    duration: 0.2,
                  },
                },
              }}
              className="overflow-hidden"
            >
              <div className="px-4 pb-3 pt-1">
                {/* Content Area */}
                <div
                  className={cn(
                    "rounded-md px-3 py-2.5 text-sm",
                    "bg-white dark:bg-purple-950/50",
                    "border border-purple-200 dark:border-purple-800",
                    "font-mono text-purple-900 dark:text-purple-100",
                    "whitespace-pre-wrap break-words"
                  )}
                >
                  {part.content || (
                    <span className="text-purple-400 dark:text-purple-600 italic">
                      Thinking in progress...
                    </span>
                  )}
                </div>

                {/* Timestamp Information (if available) */}
                {part.startTime && (
                  <div className="mt-2 text-xs text-purple-600 dark:text-purple-400 font-mono">
                    Started: {new Date(part.startTime).toLocaleTimeString()}
                    {part.endTime && (
                      <span className="ml-3">
                        Ended: {new Date(part.endTime).toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          </CollapsibleContent>
        )}
      </AnimatePresence>
    </Collapsible>
  );
}

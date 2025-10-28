/**
 * Utility functions for parsing XML-style tags from LLM responses
 * Specifically handles <think>...</think> tags in streaming content
 */

export interface ThinkingBlock {
  content: string;
  startIndex: number;
  endIndex: number;
}

export interface ParsedThinkingContent {
  thinkingBlocks: ThinkingBlock[];
  cleanText: string;
  hasIncompleteTag: boolean;
}

/**
 * Extracts <think>...</think> blocks from text and returns cleaned text
 * Handles streaming scenarios where tags might be incomplete
 *
 * @param text - Raw text possibly containing <think> tags
 * @returns Object with thinking blocks, cleaned text, and incomplete tag status
 */
export function extractThinkTags(text: string): ParsedThinkingContent {
  const thinkingBlocks: ThinkingBlock[] = [];
  let cleanText = text;
  let hasIncompleteTag = false;

  // Regex to match complete <think>...</think> blocks
  // Uses non-greedy matching to handle multiple blocks
  const completeTagRegex = /<think>([\s\S]*?)<\/think>/g;

  let match;
  let lastIndex = 0;
  const segments: string[] = [];

  // Extract all complete thinking blocks
  while ((match = completeTagRegex.exec(text)) !== null) {
    thinkingBlocks.push({
      content: match[1].trim(),
      startIndex: match.index,
      endIndex: match.index + match[0].length,
    });

    // Add text before this thinking block to segments
    segments.push(text.substring(lastIndex, match.index));
    lastIndex = match.index + match[0].length;
  }

  // Add remaining text after last thinking block
  if (lastIndex < text.length) {
    segments.push(text.substring(lastIndex));
  }

  cleanText = segments.join("");

  // Check for incomplete opening tag (streaming scenario)
  // Match partial <think> or <think>content without closing tag
  const incompleteOpenTag = /<think(?:>[\s\S]*)?$/;
  if (incompleteOpenTag.test(text)) {
    hasIncompleteTag = true;
  }

  return {
    thinkingBlocks,
    cleanText,
    hasIncompleteTag,
  };
}

/**
 * Checks if text ends with an incomplete <think> tag
 * Useful for buffering during streaming
 *
 * @param text - Text to check
 * @returns True if there's an incomplete opening tag
 */
export function isThinkTagOpen(text: string): boolean {
  // Remove all complete tags first
  const withoutCompleteTags = text.replace(/<think>[\s\S]*?<\/think>/g, "");

  // Check if there's an opening tag without a closing tag
  const openTagCount = (withoutCompleteTags.match(/<think>/g) || []).length;
  const closeTagCount = (withoutCompleteTags.match(/<\/think>/g) || []).length;

  return openTagCount > closeTagCount;
}

/**
 * Extracts thinking content from a buffer of accumulated text
 * Used during streaming to progressively extract thinking blocks
 *
 * @param buffer - Accumulated text buffer
 * @returns Object with extracted thinking content and remaining buffer
 */
export function extractStreamingThinking(buffer: string): {
  thinking: string;
  remainingBuffer: string;
  isComplete: boolean;
} {
  // Look for complete thinking blocks
  const completeMatch = buffer.match(/<think>([\s\S]*?)<\/think>/);

  if (completeMatch) {
    const thinking = completeMatch[1].trim();
    const remainingBuffer = buffer.substring(
      completeMatch.index! + completeMatch[0].length
    );

    return {
      thinking,
      remainingBuffer,
      isComplete: true,
    };
  }

  // Check for incomplete thinking block being streamed
  const incompleteMatch = buffer.match(/<think>([\s\S]*)$/);

  if (incompleteMatch) {
    return {
      thinking: incompleteMatch[1], // Content so far
      remainingBuffer: buffer, // Keep buffer intact
      isComplete: false,
    };
  }

  // No thinking content found
  return {
    thinking: "",
    remainingBuffer: buffer,
    isComplete: false,
  };
}

/**
 * Sanitizes thinking content for display
 * Removes extra whitespace and normalizes line breaks
 *
 * @param content - Raw thinking content
 * @returns Cleaned content
 */
export function sanitizeThinkingContent(content: string): string {
  return content
    .trim()
    .replace(/\n{3,}/g, "\n\n") // Max 2 consecutive line breaks
    .replace(/^\s+|\s+$/gm, ""); // Trim each line
}

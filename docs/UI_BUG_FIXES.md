# Llama Stack UI Bug Fixes

This document details two critical UI bugs identified and fixed in the Llama Stack chat playground interface.

## Bug Fix 1: Agent Instructions Overflow

### Problem Description

The Agent Instructions field in the chat playground settings panel was overflowing its container, causing text to overlap with the "Agent Tools" section below. This occurred when agent instructions exceeded the fixed height container (96px / h-24).

**Symptoms:**
- Long instruction text overflowed beyond the container boundaries
- Text overlapped with "Agent Tools" section
- No scrolling mechanism available
- Poor user experience when viewing lengthy instructions

**Location:** `llama_stack/ui/app/chat-playground/page.tsx:1467`

### Root Cause

The Agent Instructions display div had:
- Fixed height (`h-24` = 96px)
- No overflow handling
- Text wrapping enabled but no scroll capability

```tsx
// BEFORE (Broken)
<div className="w-full h-24 px-3 py-2 text-sm border border-input rounded-md bg-muted text-muted-foreground">
  {instructions}
</div>
```

### Solution

Added `overflow-y-auto` to enable vertical scrolling when content exceeds the fixed height.

```tsx
// AFTER (Fixed)
<div className="w-full h-24 px-3 py-2 text-sm border border-input rounded-md bg-muted text-muted-foreground overflow-y-auto">
  {instructions}
</div>
```

**Changes:**
- File: `llama_stack/ui/app/chat-playground/page.tsx`
- Line: 1467
- Change: Added `overflow-y-auto` to className

### Benefits

- Text wraps naturally within container
- Scrollbar appears automatically when needed
- No overlap with sections below
- Maintains read-only design intent
- Improved user experience

---

## Bug Fix 2: Duplicate Content in Chat Responses

### Problem Description

Chat assistant responses were appearing twice within a single message bubble. The content would stream in correctly, then duplicate itself at the end of the response, resulting in confusing and unprofessional output.

**Symptoms:**
- Content appeared once during streaming
- Same content duplicated after stream completion
- Duplication occurred within single message bubble
- Affected all assistant responses during streaming

**Example from logs:**
```
Response:
  <think>reasoning</think>
  Answer content here
  <think>reasoning</think>  ← DUPLICATE
  Answer content here        ← DUPLICATE
```

**Location:** `llama_stack/ui/app/chat-playground/page.tsx:790-1094`

### Root Cause Analysis

The streaming API sends two types of events:

1. **Delta chunks** (incremental):
   - "Hello"
   - " world"
   - "!"
   - Accumulated: `fullContent = "Hello world!"`

2. **turn_complete event** (final):
   - Contains the **complete accumulated content**
   - Sent after streaming finishes

**The Bug:** The `processChunk` function was extracting text from both:
- Streaming deltas (lines 790-1025) ✅
- `turn_complete` event's `turn.output_message.content` (lines 930-942) ❌

This caused the accumulated content to be **appended again** to `fullContent`, resulting in duplication.

### Solution

Added an early return in `processChunk` to skip `turn_complete` events entirely, since we already have the complete content from streaming deltas.

```tsx
// AFTER (Fixed) - Added at line 795
const processChunk = (
  chunk: unknown
): { text: string | null; isToolCall: boolean } => {
  const chunkObj = chunk as Record<string, unknown>;

  // Skip turn_complete events to avoid duplicate content
  // These events contain the full accumulated content which we already have from streaming deltas
  if (
    chunkObj?.event &&
    typeof chunkObj.event === "object" &&
    chunkObj.event !== null
  ) {
    const event = chunkObj.event as Record<string, unknown>;
    if (
      event?.payload &&
      typeof event.payload === "object" &&
      event.payload !== null
    ) {
      const payload = event.payload as Record<string, unknown>;
      if (payload.event_type === "turn_complete") {
        return { text: null, isToolCall: false };
      }
    }
  }

  // ... rest of function continues
}
```

**Changes:**
- File: `llama_stack/ui/app/chat-playground/page.tsx`
- Lines: 795-813 (new code block)
- Change: Added early return check for `turn_complete` events

### Validation

Tested with actual log file from `/home/asallas/workarea/logs/applications/llama-stack/llm_req_res.log` which showed:
- Original response (lines 5-62)
- Duplicate content (lines 63-109)

After fix:
- Only original response appears once
- No duplication
- All content types work correctly (text, code blocks, thinking blocks)

### Benefits

- Clean, professional responses
- No confusing duplicate content
- Maintains all functionality (tool calls, RAG, etc.)
- Improved user experience
- Validates streaming architecture understanding

---

## Testing Performed

### Agent Instructions Overflow
✅ Tested with short instructions (no scrollbar needed)
✅ Tested with long instructions (scrollbar appears)
✅ Verified no overlap with sections below
✅ Confirmed read-only behavior maintained

### Duplicate Content Fix
✅ Tested with simple text responses
✅ Tested with multi-paragraph responses
✅ Tested with code blocks
✅ Tested with thinking blocks (`<think>`)
✅ Tested with tool calls
✅ Tested with RAG queries
✅ Validated with production log files

---

## Files Modified

1. `llama_stack/ui/app/chat-playground/page.tsx`
   - Line 1467: Added `overflow-y-auto` for Agent Instructions
   - Lines 795-813: Added `turn_complete` event filtering

---

## Related Documentation

- Chat Playground Architecture: `llama_stack/ui/app/chat-playground/`
- Message Components: `llama_stack/ui/components/chat-playground/`
- API Integration: Llama Stack Agents API

---

## Future Considerations

### Agent Instructions
- Consider making instructions editable after creation (requires API change)
- Add copy-to-clipboard button for long instructions
- Implement instruction templates

### Streaming Architecture
- Monitor for other event types that might cause similar issues
- Add debug mode to log event types during streaming
- Consider telemetry for streaming errors

---

## Impact

**User Experience:**
- ✅ Professional, clean chat interface
- ✅ No confusing duplicate content
- ✅ Better handling of long agent instructions
- ✅ Improved reliability

**Code Quality:**
- ✅ Better understanding of streaming event flow
- ✅ More robust event handling
- ✅ Clear separation of delta vs final events

**Maintenance:**
- ✅ Well-documented fixes
- ✅ Clear root cause understanding
- ✅ Testable and verifiable

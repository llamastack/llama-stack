export interface TextContentBlock {
  type: "text";
  text: string;
}

export interface ImageUrlDetail {
  url: string;
  detail?: "low" | "high" | "auto";
}

export interface ImageUrlContentBlock {
  type: "image_url";
  // Support both simple URL string and detailed object, though our parser currently just looks for type: "image_url"
  image_url: string | ImageUrlDetail;
}

// Union of known content part types. Add more specific types as needed.
export type ChatMessageContentPart =
  | TextContentBlock
  | ImageUrlContentBlock
  | { type: string; [key: string]: unknown }; // Fallback for other potential types

export interface ChatMessage {
  role: string;
  content: string | ChatMessageContentPart[]; // Updated content type
  name?: string | null;
  tool_calls?: unknown | null; // This could also be refined to a more specific ToolCall[] type
}

export interface Choice {
  message: ChatMessage;
  finish_reason: string;
  index: number;
  logprobs?: unknown | null;
}

export interface ChatCompletion {
  id: string;
  choices: Choice[];
  object: string;
  created: number;
  model: string;
  input_messages: ChatMessage[];
}

export interface ListChatCompletionsResponse {
  data: ChatCompletion[];
  has_more: boolean;
  first_id: string;
  last_id: string;
  object: "list";
}

export type PaginationStatus = "idle" | "loading" | "loading-more" | "error";

export interface PaginationState {
  data: ChatCompletion[];
  status: PaginationStatus;
  hasMore: boolean;
  error: Error | null;
  lastId: string | null;
}

export interface UsePaginationOptions {
  /** Number of items to load per page (default: 20) */
  limit?: number;
  /** Filter by specific model */
  model?: string;
  /** Sort order for results (default: "desc") */
  order?: "asc" | "desc";
}

// Response types for OpenAI Responses API
export interface ResponseInputMessageContent {
  text?: string;
  type: "input_text" | "input_image" | "output_text";
  image_url?: string;
  detail?: "low" | "high" | "auto";
}

export interface ResponseMessage {
  content: string | ResponseInputMessageContent[];
  role: "system" | "developer" | "user" | "assistant";
  type: "message";
  id?: string;
  status?: string;
}

export interface ResponseToolCall {
  id: string;
  status: string;
  type: "web_search_call" | "function_call";
  arguments?: string;
  call_id?: string;
  name?: string;
}

export type ResponseOutput = ResponseMessage | ResponseToolCall;

export interface ResponseInput {
  type: string;
  content?: string | ResponseInputMessageContent[];
  role?: string;
  [key: string]: unknown; // Flexible for various input types
}

export interface OpenAIResponse {
  id: string;
  created_at: number;
  model: string;
  object: "response";
  status: string;
  output: ResponseOutput[];
  input: ResponseInput[];
  error?: {
    code: string;
    message: string;
  };
  parallel_tool_calls?: boolean;
  previous_response_id?: string;
  temperature?: number;
  top_p?: number;
  truncation?: string;
  user?: string;
}

export interface InputItemListResponse {
  data: ResponseInput[];
  object: "list";
}

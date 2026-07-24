export const TOOL_CATEGORY_IDS = [
  "llm-security",
  "model-scanning",
  "adversarial-ml",
  "supply-chain",
  "agent-security",
  "privacy",
] as const;

export type ToolCategoryId = (typeof TOOL_CATEGORY_IDS)[number];
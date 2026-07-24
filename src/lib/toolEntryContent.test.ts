import { describe, expect, it } from "vitest";
import { CATEGORY_BY_ID } from "../data/categories";
import { createToolEntryContent } from "./toolEntryContent";

const category = CATEGORY_BY_ID.get("llm-security");

if (!category) {
  throw new Error("Missing llm-security category fixture");
}

describe("createToolEntryContent", () => {
  it("creates a concise source-bounded answer from verified catalog fields", () => {
    const content = createToolEntryContent(
      {
        name: "Garak",
        url: "https://github.com/NVIDIA/garak",
        description: "LLM vulnerability scanner",
      },
      category,
    );

    const wordCount = content.answer.split(/\s+/).length;
    expect(content.answer).toContain("Garak is included in the Awesome MLSecOps");
    expect(content.answer).toContain("“LLM vulnerability scanner.”");
    expect(content.answer).toContain("NVIDIA/garak repository on GitHub");
    expect(content.answer).toContain("CI and reporting support");
    expect(content.answer).toContain("not a certification, comparative ranking, or endorsement");
    expect(wordCount).toBeGreaterThanOrEqual(110);
    expect(wordCount).toBeLessThanOrEqual(170);
  });

  it("keeps unique search metadata within practical length limits", () => {
    const content = createToolEntryContent(
      {
        name: "A deliberately long security resource name used for metadata validation",
        url: "https://example.com/security-resource",
        description: "A deliberately detailed resource description used to verify safe metadata truncation behavior",
      },
      category,
    );

    expect(content.title.length).toBeLessThanOrEqual(60);
    expect(content.metaDescription.length).toBeLessThanOrEqual(158);
    expect(content.sourceStatement).toBe("The linked first-party source is published at example.com.");
  });
});

export const OG_PAGES = {
  home: { title: "Awesome MLSecOps", description: "Curated ML and AI security tools, guides, jobs, and research." },
  "what-is-mlsecops": { title: "What Is MLSecOps?", description: "A practical definition, lifecycle, and implementation guide." },
  faq: { title: "MLSecOps FAQ", description: "Direct answers about machine learning and AI security operations." },
  tools: { title: "MLSecOps Tool Evaluation Guides", description: "Decision criteria and source-linked resources for six machine learning and AI security problems." },
  "tools/llm-security": { title: "LLM Security Tool Evaluation Guide", description: "Evaluate red-team tools, scanners, guardrails, and LLM security testing resources." },
  "tools/model-scanning": { title: "ML Model Scanner Evaluation Guide", description: "Evaluate model artifact scanners, validation tools, formats, evidence, and CI support." },
  "tools/adversarial-ml": { title: "Adversarial ML Tool Evaluation Guide", description: "Evaluate attack, defense, robustness, and reproducibility resources for adversarial ML." },
  "tools/supply-chain": { title: "AI Supply Chain Tool Evaluation Guide", description: "Evaluate model provenance, signing, ML-BOM, registry, and pipeline security resources." },
  "tools/agent-security": { title: "AI Agent Security Tool Evaluation Guide", description: "Evaluate agent permissions, memory, sandboxing, approvals, and MCP security resources." },
  "tools/privacy": { title: "Privacy ML Tool Evaluation Guide", description: "Evaluate differential privacy, anonymization, encrypted computation, and privacy testing tools." },
  "mlsecops-vs-devsecops": { title: "MLSecOps vs DevSecOps", description: "Compare responsibilities, controls, assets, and operating models." },
  jobs: { title: "AI and ML Security Jobs", description: "Curated current roles in AI safety, red teaming, and ML security." },
  // sponsor: { title: "Sponsor Awesome MLSecOps", description: "Reach security and ML practitioners through transparent sponsorship." },
  newsletter: { title: "The MLSecOps Hacker Newsletter", description: "Subscribe and browse the on-domain MLSecOps newsletter archive." },
} as const;

export type OgPageKey = keyof typeof OG_PAGES;
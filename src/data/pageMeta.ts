export const OG_PAGES = {
  home: { title: "Awesome MLSecOps", description: "Curated ML and AI security tools, guides, jobs, and research." },
  "what-is-mlsecops": { title: "What Is MLSecOps?", description: "A practical definition, lifecycle, and implementation guide." },
  faq: { title: "MLSecOps FAQ", description: "Direct answers about machine learning and AI security operations." },
  tools: { title: "MLSecOps Tools", description: "Browse curated tools by machine learning and AI security category." },
  "tools/llm-security": { title: "Open-Source LLM Security Tools", description: "Scanners, red-team harnesses, guardrails, and testing tools." },
  "tools/model-scanning": { title: "ML Model Scanning Tools", description: "Tools for model artifacts, unsafe serialization, and validation." },
  "tools/adversarial-ml": { title: "Adversarial ML Tools", description: "Libraries and frameworks for attacks, defenses, and robustness." },
  "tools/supply-chain": { title: "AI Supply-Chain Security Tools", description: "Model provenance, signing, SBOM, and pipeline security resources." },
  "tools/agent-security": { title: "AI Agent and MCP Security Tools", description: "Secure agent memory, permissions, tools, runtimes, and MCP servers." },
  "tools/privacy": { title: "Privacy-Preserving ML Tools", description: "Differential privacy, anonymization, and privacy testing resources." },
  "mlsecops-vs-devsecops": { title: "MLSecOps vs DevSecOps", description: "Compare responsibilities, controls, assets, and operating models." },
  jobs: { title: "AI and ML Security Jobs", description: "Curated current roles in AI safety, red teaming, and ML security." },
  // sponsor: { title: "Sponsor Awesome MLSecOps", description: "Reach security and ML practitioners through transparent sponsorship." },
  newsletter: { title: "The MLSecOps Hacker Newsletter", description: "Subscribe and browse the on-domain MLSecOps newsletter archive." },
} as const;

export type OgPageKey = keyof typeof OG_PAGES;
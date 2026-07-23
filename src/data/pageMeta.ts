export const OG_PAGES = {
  home: { title: "Awesome MLSecOps", description: "Curated ML and AI security tools, guides, jobs, and research." },
  "what-is-mlsecops": { title: "What Is MLSecOps?", description: "A practical definition, lifecycle, and implementation guide." },
  faq: { title: "MLSecOps FAQ", description: "Direct answers about machine learning and AI security operations." },
  tools: { title: "MLSecOps Tools and Resources", description: "Browse curated resources by machine learning and AI security category." },
  "tools/llm-security": { title: "LLM Security Tools and Resources", description: "Scanners, red-team harnesses, guardrails, research, and testing tools." },
  "tools/model-scanning": { title: "ML Model Scanning Resources", description: "Tools and research for model artifacts, unsafe serialization, and validation." },
  "tools/adversarial-ml": { title: "Adversarial ML Resources", description: "Libraries, research, and frameworks for attacks, defenses, and robustness." },
  "tools/supply-chain": { title: "AI Supply-Chain Security Resources", description: "Model provenance, signing, SBOM, and pipeline security resources." },
  "tools/agent-security": { title: "AI Agent and MCP Security Resources", description: "Resources for agent memory, permissions, tools, runtimes, and MCP servers." },
  "tools/privacy": { title: "Privacy-Preserving ML Resources", description: "Differential privacy, anonymization, and privacy testing resources." },
  "mlsecops-vs-devsecops": { title: "MLSecOps vs DevSecOps", description: "Compare responsibilities, controls, assets, and operating models." },
  jobs: { title: "AI and ML Security Jobs", description: "Curated current roles in AI safety, red teaming, and ML security." },
  // sponsor: { title: "Sponsor Awesome MLSecOps", description: "Reach security and ML practitioners through transparent sponsorship." },
  newsletter: { title: "The MLSecOps Hacker Newsletter", description: "Subscribe and browse the on-domain MLSecOps newsletter archive." },
} as const;

export type OgPageKey = keyof typeof OG_PAGES;
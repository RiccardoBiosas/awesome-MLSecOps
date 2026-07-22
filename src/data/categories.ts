import { REPOSITORY_URL } from "../config/site";
import type { ToolCategoryId } from "../lib/toolCategories";

export type PublicCategoryId = ToolCategoryId;

export type ToolCategory = {
  id: PublicCategoryId;
  name: string;
  shortName: string;
  query: string;
  metaDescription: string;
  shortDescription: string;
  intro: [string, string];
  evaluation: string[];
  attackVectors: Array<{ name: string; url: string }>;
  readmeUrl: string;
};

export const TOOL_CATEGORIES: ToolCategory[] = [
  {
    id: "llm-security",
    name: "LLM Security and Red Teaming",
    shortName: "LLM security",
    query: "open-source LLM security tools",
    metaDescription: "Open-source LLM security tools for prompt injection testing, AI red teaming, guardrails, vulnerability scanning, and evaluation.",
    shortDescription: "Test prompts, model behavior, guardrails, and application controls against abuse.",
    intro: [
      "LLM security tools help teams find weaknesses that conventional application scanners cannot see. They exercise model behavior, system prompts, retrieval pipelines, tool calls, output handling, and policy controls under adversarial input. Common uses include prompt-injection testing, jailbreak evaluation, sensitive-data leakage checks, guardrail validation, and repeatable red-team campaigns. The tools listed here come directly from the community-maintained Awesome MLSecOps catalog rather than a paid placement program.",
      "Evaluate an LLM security tool by matching its test library to your threat model and deployment architecture. Check whether it supports the models, APIs, RAG systems, and agent frameworks you use. Prefer reproducible test cases, machine-readable results, CI integration, clear success criteria, and controls for handling sensitive prompts. A broad attack library is useful, but evidence quality matters more than raw test counts. Review maintenance activity and licensing before putting any scanner in a production pipeline.",
    ],
    evaluation: ["Threat and model coverage", "Reproducible evaluations", "CI and reporting support", "Sensitive-data handling"],
    attackVectors: [
      { name: "Prompt injection attacks", url: "https://research.nccgroup.com/2022/12/05/exploring-prompt-injection-attacks" },
      { name: "Excessive agent permissions", url: "https://genai.owasp.org/llmrisk/llm08-excessive-agency/" },
      { name: "Model denial of service", url: "https://genai.owasp.org/llmrisk/llm04-model-denial-of-service/" },
    ],
    readmeUrl: `${REPOSITORY_URL}#open-source-security-tools`,
  },
  {
    id: "model-scanning",
    name: "Model Scanning and Validation",
    shortName: "Model scanning",
    query: "ML model scanning tools",
    metaDescription: "ML model scanning tools for unsafe serialization, malicious artifacts, vulnerabilities, validation, and machine learning security checks.",
    shortDescription: "Inspect model files, notebooks, code, and behavior before release or deployment.",
    intro: [
      "Model scanning tools inspect machine learning artifacts and surrounding code before those assets enter trusted environments. A scanner may detect unsafe serialization instructions, embedded payloads, vulnerable dependencies, suspicious operators, policy violations, or unexpected model behavior. This category also includes validation tools that establish whether a model still meets security and quality expectations after a training, conversion, or packaging step.",
      "Choose a scanner based on the artifact formats and frameworks in your supply chain, not on a generic claim of AI coverage. Verify how it handles Pickle-derived formats, SafeTensors, ONNX, notebooks, and custom packages. Review whether findings identify concrete evidence and support automation through exit codes, SARIF, APIs, or CI integrations. Scanning is one control rather than proof of safety: provenance, signatures, isolated loading, access control, and behavioral testing remain necessary around it.",
    ],
    evaluation: ["Supported artifact formats", "Detection evidence", "False-positive handling", "CI and SARIF output"],
    attackVectors: [
      { name: "AI supply-chain attacks", url: "https://owasp.org/www-project-machine-learning-security-top-10/docs/ML06_2023-AI_Supply_Chain_Attacks" },
      { name: "Models as executable code", url: "https://hiddenlayer.com/research/models-are-code/" },
    ],
    readmeUrl: `${REPOSITORY_URL}#ml-code-security`,
  },
  {
    id: "adversarial-ml",
    name: "Adversarial Machine Learning",
    shortName: "Adversarial ML",
    query: "adversarial machine learning tools",
    metaDescription: "Adversarial machine learning tools for evasion, poisoning, model extraction, inversion, robustness testing, and defensive research.",
    shortDescription: "Evaluate evasion, poisoning, extraction, inversion, and model robustness.",
    intro: [
      "Adversarial machine learning tools let researchers and defenders test how models behave when an attacker manipulates inputs, training data, model access, or surrounding assumptions. The category spans evasion examples, poisoning and backdoor research, model extraction, inversion, membership inference, and robustness measurement. Some projects are production-oriented libraries; others are research artifacts intended for controlled experiments and should be treated accordingly.",
      "Start with a specific attacker capability and measurable security question. Confirm that the tool supports your modality, framework, model-access level, and threat constraints. Look for documented baselines, deterministic experiment configuration, maintained datasets, and metrics that distinguish model accuracy from security robustness. Offensive capabilities should run only in authorized environments with isolated data. Before adopting a defense, check whether evaluations include adaptive attackers rather than only the attack used to design the control.",
    ],
    evaluation: ["Threat-model fit", "Modality and framework support", "Adaptive attack evaluation", "Experiment reproducibility"],
    attackVectors: [
      { name: "Data poisoning", url: "https://github.com/ch-shin/awesome-data-poisoning" },
      { name: "Model inversion", url: "https://blogs.rstudio.com/ai/posts/2020-05-15-model-inversion-attacks/" },
      { name: "Model evasion", url: "https://www.ibm.com/docs/en/watsonx/saas?topic=atlas-evasion-attack" },
    ],
    readmeUrl: `${REPOSITORY_URL}#attack-vectors`,
  },
  {
    id: "supply-chain",
    name: "AI Supply-Chain Security",
    shortName: "Supply-chain security",
    query: "AI supply chain security tools",
    metaDescription: "AI supply-chain security tools for model provenance, signing, ML-BOMs, unsafe artifacts, dependencies, and secure MLOps pipelines.",
    shortDescription: "Protect model provenance, artifacts, dependencies, signing, and delivery pipelines.",
    intro: [
      "AI supply-chain security covers the path from data and training code to packaged models, registries, deployment images, and inference services. Attackers can compromise dependencies, tamper with artifacts, exploit unsafe model formats, poison shared assets, or substitute an untrusted model during delivery. The tools in this category support controls such as model signing, provenance records, artifact inspection, ML bills of materials, and policy enforcement in MLOps infrastructure.",
      "Evaluate coverage across the supply chain you actually operate. A useful tool should identify artifacts with stable hashes, integrate with build and registry workflows, and produce evidence that downstream systems can verify. Check support for CycloneDX, SPDX, SLSA, Sigstore, and the model formats your teams exchange. Strong controls also define trust roots, key management, exception handling, and incident response. An inventory without verification is useful for visibility, but it does not by itself prevent artifact substitution or malicious loading.",
    ],
    evaluation: ["Provenance and signing support", "ML-BOM formats", "Registry and CI integration", "Policy enforcement"],
    attackVectors: [
      { name: "ML supply-chain attacks", url: "https://owasp.org/www-project-machine-learning-security-top-10/docs/ML06_2023-AI_Supply_Chain_Attacks" },
      { name: "ClearML supply-chain research", url: "https://hiddenlayer.com/research/not-so-clear-how-mlops-solutions-can-muddy-the-waters-of-your-supply-chain/" },
      { name: "Model conversion risks", url: "https://hiddenlayer.com/research/silent-sabotage/" },
    ],
    readmeUrl: `${REPOSITORY_URL}#mlops-infrastructure-vulnerabilities`,
  },
  {
    id: "agent-security",
    name: "AI Agent and MCP Security",
    shortName: "Agent security",
    query: "AI agent security tools",
    metaDescription: "AI agent and MCP security tools for memory, tool permissions, identity, sandboxing, prompt injection, monitoring, and red teaming.",
    shortDescription: "Secure agent tools, memory, identity, permissions, sandboxes, and MCP servers.",
    intro: [
      "AI agent security focuses on systems that can choose actions, call tools, retain memory, and interact with external services. These capabilities turn prompt injection and untrusted context into authorization and execution risks. Model Context Protocol servers add another integration boundary involving tool descriptions, credentials, transport, and user approval. Tools in this category inspect agent workflows, scan MCP or skill definitions, protect memory, test excessive agency, and isolate runtime actions.",
      "Assess agent-security tooling against the permissions and data paths in your architecture. Look for explicit identities, least-privilege scopes, per-tool policy, credential isolation, network controls, durable audit records, and human approval for consequential actions. A scanner should cover indirect prompt injection and malicious tool metadata, not only user prompts. Runtime defenses need clear failure behavior when a policy engine, sandbox, or model is unavailable. Test the complete action path because a safe model response can still trigger an unsafe downstream tool call.",
    ],
    evaluation: ["Tool-level authorization", "Memory and context controls", "Sandbox boundaries", "Audit and approval workflows"],
    attackVectors: [
      { name: "Excessive agency", url: "https://genai.owasp.org/llmrisk/llm08-excessive-agency/" },
      { name: "Prompt injection", url: "https://research.nccgroup.com/2022/12/05/exploring-prompt-injection-attacks" },
      { name: "Agent memory poisoning defense", url: "https://github.com/OWASP/www-project-agent-memory-guard" },
    ],
    readmeUrl: `${REPOSITORY_URL}#open-source-security-tools`,
  },
  {
    id: "privacy",
    name: "Privacy-Preserving Machine Learning",
    shortName: "ML privacy",
    query: "privacy preserving machine learning tools",
    metaDescription: "Privacy-preserving machine learning tools for differential privacy, anonymization, encrypted computation, and privacy attack testing.",
    shortDescription: "Reduce sensitive-data exposure and test privacy leakage in ML systems.",
    intro: [
      "Privacy-preserving machine learning tools reduce or measure the exposure of sensitive information across data preparation, training, evaluation, and inference. Techniques include differential privacy, anonymization, encrypted computation, privacy auditing, and tests for membership inference or model inversion. These controls address different risks: anonymized inputs do not guarantee a trained model cannot memorize records, while a privacy budget does not secure the surrounding data pipeline.",
      "Select tools by defining the protected data, attacker access, acceptable utility loss, and compliance evidence you need. For differential privacy, inspect accounting methods, clipping behavior, composition, and how privacy budgets are reported. For anonymization, evaluate re-identification risk and whether visual or tabular outputs retain hidden metadata. Privacy attack tools should reproduce realistic access conditions. Favor projects with clear threat assumptions, measurable guarantees, documented limitations, and integration points that keep privacy checks repeatable as models and datasets change.",
    ],
    evaluation: ["Explicit privacy guarantees", "Utility and accuracy impact", "Re-identification testing", "Repeatable privacy accounting"],
    attackVectors: [
      { name: "Membership inference", url: "https://arxiv.org/abs/2103.07853" },
      { name: "Model inversion", url: "https://blogs.rstudio.com/ai/posts/2020-05-15-model-inversion-attacks/" },
      { name: "Gradient leakage", url: "https://ieeexplore.ieee.org/document/10107713" },
    ],
    readmeUrl: `${REPOSITORY_URL}#data`,
  },
];

export const CATEGORY_BY_ID = new Map(TOOL_CATEGORIES.map((category) => [category.id, category]));
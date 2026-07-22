export type FaqItem = { question: string; answer: string };

export const WHAT_IS_FAQS: FaqItem[] = [
  {
    question: "What does MLSecOps stand for?",
    answer: "MLSecOps stands for Machine Learning Security Operations. It applies security engineering, testing, supply-chain controls, monitoring, and incident response across the machine-learning lifecycle.",
  },
  {
    question: "How is MLSecOps different from MLOps?",
    answer: "MLOps makes model delivery reliable and repeatable. MLSecOps adds explicit protection against adversarial attacks, unsafe artifacts, privacy leakage, model theft, compromised pipelines, and abuse after deployment.",
  },
  {
    question: "Is LLM security part of MLSecOps?",
    answer: "Yes. MLSecOps covers LLM applications, including prompt injection, jailbreaks, sensitive-data exposure, RAG poisoning, insecure tool use, guardrail testing, and AI agent security.",
  },
  {
    question: "Who owns MLSecOps?",
    answer: "MLSecOps is a shared operating responsibility. Security engineers define controls and testing, ML engineers secure models and data, platform teams protect pipelines, and product owners manage acceptable risk and response decisions.",
  },
  {
    question: "Where should a team start with MLSecOps?",
    answer: "Start with an inventory of models, data, pipelines, tools, and external dependencies. Threat-model the highest-impact system, add artifact and access controls, establish repeatable security tests, and define monitoring and incident ownership before scaling the program.",
  },
];

export const FAQ_ITEMS: FaqItem[] = [
  ...WHAT_IS_FAQS,
  {
    question: "What tools are used for MLSecOps?",
    answer: "MLSecOps tools include model and artifact scanners, adversarial ML libraries, LLM red-team harnesses, guardrails, privacy toolkits, provenance and signing systems, ML-BOM generators, agent-security scanners, and security evaluation benchmarks.",
  },
  {
    question: "Does MLSecOps replace DevSecOps?",
    answer: "No. MLSecOps extends DevSecOps for risks created by data, models, training, inference, and probabilistic behavior. Teams still need conventional software, cloud, identity, dependency, and infrastructure security controls.",
  },
  {
    question: "What is an ML supply-chain attack?",
    answer: "An ML supply-chain attack compromises a dependency, dataset, model artifact, registry, serialization format, build process, or deployment path so an untrusted component reaches a machine-learning system.",
  },
  {
    question: "What is adversarial machine learning?",
    answer: "Adversarial machine learning studies attacks and defenses involving evasion, poisoning, backdoors, model extraction, inversion, membership inference, and other attempts to manipulate or learn from model behavior.",
  },
  {
    question: "How should AI agents be secured?",
    answer: "Secure AI agents with explicit identities, least-privilege tool permissions, isolated credentials, constrained networks, memory controls, sandboxed execution, durable logs, and human approval for consequential actions.",
  },
  {
    question: "How is MLSecOps measured?",
    answer: "Useful MLSecOps measures include asset coverage, signed-artifact coverage, security-test pass rates, unresolved high-risk findings, privacy-budget compliance, time to detect model abuse, and time to contain AI-related incidents.",
  },
  {
    question: "Is MLSecOps only for large organizations?",
    answer: "No. Small teams can begin with a model inventory, threat model, approved artifact formats, dependency and model scanning, restricted service accounts, structured red-team tests, and a simple incident-response owner.",
  },
];
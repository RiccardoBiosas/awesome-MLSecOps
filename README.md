# Awesome MLSecOps 🛡️🤖

## What is MLSecOps?

**MLSecOps (Machine Learning Security Operations)** is the practice of integrating security throughout the machine learning lifecycle—from data collection and model development to deployment, monitoring, and incident response. It applies security testing, threat modeling, supply-chain protection, access controls, and continuous monitoring to machine learning models, MLOps pipelines, LLM applications, and AI agents.

This curated catalog helps security engineers, ML practitioners, developers, and AI red teams discover open-source MLSecOps tools, adversarial machine learning research, AI security frameworks, threat-modeling resources, and practical learning materials.

⭐ If this catalog is useful, [star the repository](https://github.com/RiccardoBiosas/awesome-MLSecOps) or read the [contribution guidelines](CONTRIBUTING.md) to suggest a resource.

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/RiccardoBiosas/awesome-MLSecOps/graphs/commit-activity)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)
[![Follow](https://img.shields.io/twitter/follow/RBiosas.svg?style=social&label=Follow)](https://twitter.com/RBiosas)

![MLSecOps Banner](https://github.com/user-attachments/assets/966affca-442a-4859-b450-774b8d48c6cc)


## Table of Contents
- [What is MLSecOps?](#what-is-mlsecops)
- [Open Source Security Tools](#open-source-security-tools)
- [Commercial MLSecOps and AI Security Tools](#commercial-mlsecops-and-ai-security-tools)
- [Data Privacy and Anonymization Tools](#data-privacy-and-anonymization-tools)
- [ML Code Security](#ml-code-security)
- [AI Security and MLSecOps Beginner Resources](#ai-security-and-mlsecops-beginner-resources)
- [Threat Modeling](#threat-modeling)
- [Attack Vectors](#attack-vectors)
- [Blogs and Publications](#blogs-and-publications)
- [MLOps Infrastructure Vulnerabilities](#mlops-infrastructure-vulnerabilities)
- [MLSecOps Pipeline](#mlsecops-pipeline)
- [Adversarial Machine Learning Research Repositories](#adversarial-machine-learning-research-repositories)
- [Community Resources](#community-resources)
- [Books](#books)
- [Infographics](#infographics)
- [Contributions](#contributions)
- [Contributors](#contributors-)
- [Repository Stats](#repository-stats)
- [Activity](#activity)
- [Support Us](#support-us)
- [License](#license)


## Open Source Security Tools

Open-source MLSecOps tools help practitioners test, monitor, and protect machine learning models, data, pipelines, LLM applications, and AI agents. The resources below include active projects and proofs of concept; evaluate maintenance status, licensing, threat coverage, and deployment suitability before production use.

### Model and Artifact Security

Model and artifact security tools detect unsafe serialization, malicious payloads, integrity failures, and other risks in machine learning model files.

| Tool | Description |
|------|-------------|
| [ModelScan](https://github.com/protectai/modelscan) | Protection Against ML Model Serialization Attacks |
| [Safetensors](https://github.com/huggingface/safetensors) | Convert pickle to a safe serialization option |

### Adversarial Machine Learning Testing

Adversarial machine learning testing tools evaluate model robustness against evasion, poisoning, extraction, inversion, and other adversarial techniques.

| Tool | Description |
|------|-------------|
| [Adversarial Robustness Toolbox](https://github.com/IBM/adversarial-robustness-toolbox) | Library of defense methods for ML models against adversarial attacks |
| [Foolbox](https://github.com/bethgelab/foolbox) | Python toolbox for creating and evaluating adversarial attacks and defenses |

### LLM Security and Red Teaming

LLM security and red-teaming tools test prompts, model behavior, application controls, and safeguards against attacks such as prompt injection and jailbreaks.

| Tool | Description |
|------|-------------|
| [Garak](https://github.com/leondz/garak) | LLM vulnerability scanner |
| [Promptfoo Scanner](https://github.com/promptfoo/promptfoo) | An open-source LLM red teaming tool |

### AI Agent Security

AI agent security tools assess memory, tools, permissions, workflows, and runtime boundaries used by autonomous and tool-enabled AI systems.

| Tool | Description |
|------|-------------|
| [Agent Memory Guard](https://github.com/OWASP/www-project-agent-memory-guard) | Official OWASP runtime defense layer that screens AI agent memory reads/writes, blocking prompt injection, secret leakage, and memory poisoning (ASI06) |
| [Agent-Wiz](https://github.com/Repello-AI/Agent-Wiz) | Python CLI by Repello AI for extracting agentic workflows from LangChain/LangGraph/CrewAI/AutoGen and running automated threat modeling |

### Privacy-Preserving Machine Learning

Privacy-preserving machine learning tools help limit sensitive-data exposure through differential privacy, encrypted computation, anonymization, and privacy testing.

| Tool | Description |
|------|-------------|
| [TensorFlow Privacy](https://github.com/tensorflow/privacy) | Library of privacy-preserving machine learning algorithms and tools |
| [OpenDP](https://github.com/opendp/opendp) | The core library of differential privacy algorithms powering the OpenDP Project |

### AI Supply-Chain Security

AI supply-chain security tools protect model artifacts, provenance, dependencies, signatures, bills of materials, and deployment pipelines.

| Tool | Description |
|------|-------------|
| [Model Transparency](https://github.com/sigstore/model-transparency) | Generate model signing metadata for provenance verification |
| [BomLens](https://github.com/sktelecom/bomlens) | Local-first SBOM generator and risk assessor that builds CycloneDX ML-BOMs for HuggingFace models with G7 minimum-elements conformance checks, plus license and known-vulnerability reports |

### Model Testing, Monitoring, and Evaluation

These projects support model validation, monitoring, robustness, safety, and comparative evaluation, including both general-purpose and security-focused assessment workflows.

| Tool | Description |
|------|-------------|
| [TensorFlow Model Analysis](https://github.com/tensorflow/model-analysis) | A library for analyzing, validating, and monitoring machine learning models in production|
| [CircleGuardBench](https://github.com/whitecircle-ai/circle-guard-bench)| A full-fledged benchmark for evaluating protection capabilities of AI models|

### Additional Open-Source Tools

The following tools and research prototypes address additional MLSecOps testing, monitoring, defense, and assurance use cases pending more granular classification.

| Tool | Description |
|------|-------------|
| [NB Defense](https://nbdefense.ai) | Secure Jupyter Notebooks |
| [MLSploit](https://github.com/mlsploit/) | Cloud framework for interactive experimentation with adversarial machine learning research |
| [Advertorch](https://github.com/BorealisAI/advertorch) | Python toolbox for adversarial robustness research |
| [Adversarial ML Threat Matrix](https://github.com/mitre/advmlthreatmatrix) | Adversarial Threat Landscape for AI Systems |
| [CleverHans](https://github.com/cleverhans-lab/cleverhans) | A library of adversarial examples and defenses for machine learning models|
| [AdvBox](https://github.com/advboxes/AdvBox) | Advbox is a toolbox to generate adversarial examples that fool neural networks in PaddlePaddle、PyTorch、Caffe2、MxNet、Keras、TensorFlow|
| [Audit AI](https://github.com/pymetrics/audit-ai) | Bias Testing for Generalized Machine Learning Applications|
| [Deep Pwning](https://github.com/cchio/deep-pwning) | Deep-pwning is a lightweight framework for experimenting with machine learning models with the goal of evaluating their robustness against a motivated adversary|
| [Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter) | An open-source library to audit data privacy in statistical and machine learning algorithms|
| [PromptInject](https://github.com/agencyenterprise/PromptInject) | A framework that assembles adversarial prompts|
| [TextAttack](https://github.com/QData/TextAttack) | TextAttack is a Python framework for adversarial attacks, data augmentation, and model training in NLP|
| [OpenAttack](https://github.com/thunlp/OpenAttack) | An Open-Source Package for Textual Adversarial Attack|
| [TextFooler](https://github.com/jind11/TextFooler) | A Model for Natural Language Attack on Text Classification and Inference|
| [Flawed Machine Learning Security](https://github.com/EthicalML/fml-security) | Practical examples of "Flawed Machine Learning Security" together with ML Security best practice across the end to end stages of the machine learning model lifecycle from training, to packaging, to deployment|
| [Adversarial Machine Learning CTF](https://github.com/arturmiller/adversarial_ml_ctf) | This repository is a CTF challenge, showing a security flaw in most (all?) common artificial neural networks. They are vulnerable for adversarial images|
| [Damn Vulnerable LLM Project](https://github.com/harishsg993010/DamnVulnerableLLMProject) | A Large Language Model designed for getting hacked|
| [Gandalf Lakera](https://gandalf.lakera.ai/) | Prompt Injection CTF playground|
| [Vigil](https://github.com/deadbits/vigil-llm) | LLM prompt injection and security scanner|
| [PALLMs (Payloads for Attacking Large Language Models)](https://github.com/mik0w/pallms) | list of various payloads for attacking LLMs collected in one place|
| [AI-exploits](https://github.com/protectai/ai-exploits) | Exploits for MLOps systems, extending beyond inputs provided to LLMs such as ChatGPT |
| [Offensive ML Playbook](https://wiki.offsecml.com/Welcome+to+the+Offensive+ML+Playbook) | Offensive ML Playbook. Notes on machine learning attacks and pentesting|
| [AnonLLM](https://github.com/fsndzomga/anonLLM) | Anonymize Personally Identifiable Information (PII) for Large Language Model APIs|
| [AI Goat](https://github.com/dhammon/ai-goat) | vulnerable LLM CTF challenges|
| [Pyrit](https://github.com/Azure/PyRIT) | The Python Risk Identification Tool for generative AI|
| [Raze to the Ground: Query-Efficient Adversarial HTML Attacks on Machine-Learning Phishing Webpage Detectors](https://github.com/advmlphish/raze_to_the_ground_aisec23) | Source code of the paper "Raze to the Ground: Query-Efficient Adversarial HTML Attacks on Machine-Learning Phishing Webpage Detectors" accepted at AISec '23|
| [Giskard](https://github.com/Giskard-AI/giskard) | Open-source testing tool for LLM applications|
| [Citadel Lens](https://www.citadel.co.jp/en/products/lens/)| Quality testing of models according to industry standards|
| [Model-Inversion-Attack-ToolBox](https://github.com/ffhibnese/Model-Inversion-Attack-ToolBox) | A framework for implementing Model Inversion attacks|
| [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) | NeMo Guardrails allow developers building LLM-based applications to add programmable guardrails between the application code and the LLM |
| [AugLy](https://github.com/facebookresearch/AugLy) | A tool for generating adversarial attacks|
| [Knockoffnets](https://github.com/tribhuvanesh/knockoffnets) | PoC to implement BlackBox attacks to steal model data|
| [Robust Intelligence Continuous Validation](https://www.robustintelligence.com/platform/continuous-validation) | Tool for continuous model validation for compliance with standards |
| [VGER](https://github.com/JosephTLucas/vger) | Jupyter Attack framework |
| [AIShield Watchtower](https://github.com/bosch-aisecurity-aishield/watchtower) | An open-source tool from AIShield for studying AI models and scanning for vulnerabilities |
| [PS-fuzz](https://github.com/prompt-security/ps-fuzz) | tool for scanning LLM vulnerabilities|
| [Mindgard CLI](https://github.com/Mindgard/cli/) | Evaluate the security of AI systems through a CLI |
| [PurpleLLama3](https://meta-llama.github.io/PurpleLlama/)| Check LLM security with Meta LLM Benchmark |
| [ARTkit](https://github.com/BCG-X-Official/artkit)|Automated prompt-based testing and evaluation of Gen AI applications|
| [LangBiTe](https://github.com/SOM-Research/LangBiTe) | A Bias Tester framework for LLMs|
| [TF-encrypted](https://tf-encrypted.io/)| Encryption for tensorflow|
| [Agentic Security](https://github.com/msoedov/agentic_security)| Agentic LLM Vulnerability Scanner / AI red teaming kit|
| [skill-audit-mcp](https://github.com/eltociear/skill-audit-mcp) | Static security scanner for MCP servers, AI agent skills, and plugins. Detects 68 attack patterns across CRITICAL/HIGH/MEDIUM/LOW — credential exfiltration, prompt injection, code execution, seed-phrase harvesting, auth bypass, path traversal. SARIF output, GitHub Action, multi-arch Docker image |
| [AIsbom](https://github.com/Lab700xOrg/aisbom) | Disassembles Pickle bytecode and parses SafeTensors/GGUF binary headers to detect malware and license risks in ML model files before load. Generates CycloneDX/SPDX SBOMs. |
| [AI-Scan-Interceptor](https://github.com/mshirakawa-ssp/ai-scan-interceptor) | Self-hostable DLP gateway for enterprise prompts to ChatGPT/Claude/Gemini (Squid + Go ICAP + mTLS, AGPL-3.0) |
| [KubeStellar Console](https://github.com/kubestellar/console) | Multi-cluster Kubernetes dashboard with MLSecOps capabilities: GPU workload monitoring, Kyverno policy enforcement, supply chain security (SBOM, SLSA), and AI/ML infrastructure observability. CNCF Sandbox project. |
| [TrustGate](https://github.com/NeuralTrust/TrustGate) | An open-source Generative Application Firewall (GAF) |
| [Whistleblower](https://github.com/Repello-AI/whistleblower) | Open-source offensive tool by Repello AI for testing LLM apps against system prompt leakage |
| [IronClaw](https://github.com/IronSecCo/ironclaw) | Security-hardened, self-hosted runtime that sandboxes autonomous AI agents in a gVisor (runsc) sandbox with network=none, a read-only rootfs, host-side credential injection, and a human-approval gateway; signed and attested supply chain (cosign, SLSA, SBOMs) |


<a id="commercial-tools"></a>
## Commercial MLSecOps and AI Security Tools

Commercial MLSecOps and AI security tools support model protection, application testing, monitoring, governance, and incident response. Evaluate technical documentation, deployment options, integrations, and independent evidence before adoption.

| Tool | Description |
|------|-------------|
| [Databricks Platform, Azure Databricks](https://azure.microsoft.com/ru-ru/products/databricks) | Datalake data management and implementation tool |
| [Hidden Layer AI Detection Response](https://hiddenlayer.com/aidr/) | Tool for detecting and responding to incidents |
| [Guardian](https://protectai.com/guardian) | Model protection in CI/CD |
| [Promptfoo](https://www.promptfoo.dev/security/) | Continuous monitoring, detection, and remediation for enterprise LLM applications |
| [NeuralTrust](https://neuraltrust.ai) | Tools to protect, secure and test GenAI Applications |


<a id="data"></a>
## Data Privacy and Anonymization Tools

Data privacy and anonymization tools help reduce exposure of personal or sensitive information in machine learning datasets, images, videos, and model interactions.

| Tool | Description |
|------|-------------|
| [ARX - Data Anonymization Tool](https://arx.deidentifier.org/) | Tool for anonymizing datasets |
| [Data-Veil](https://veil.ai/) | Data masking and anonymization tool |
| [Tool for IMG anonymization](https://github.com/PacktPublishing/Adversarial-AI---Attacks-Mitigations-and-Defense-Strategies/blob/main/ch10/notebooks/Image%20Anonymization.ipynb)| Image anonymization|
| [Tool for DATA anonymization](https://github.com/PacktPublishing/Adversarial-AI---Attacks-Mitigations-and-Defense-Strategies/blob/main/ch10/notebooks/Data%20Anonymization.ipynb)| Data anonymization|
| [BMW Anonymization API](https://github.com/BMW-InnovationLab/BMW-Anonymization-API) | This repository allows you to anonymize sensitive information in images/videos. The solution is fully compatible with the DL-based training/inference solutions that we already published/will publish for Object Detection and Semantic Segmentation |
| [DeepPrivacy2](https://github.com/hukkelas/deep_privacy2)| A Toolbox for Realistic Image Anonymization |
| [PPAP](https://github.com/tgisaturday/PPAP)|Latent-space-level Image Anonymization with Adversarial Protector Networks|

## ML Code Security

ML code security resources help practitioners identify vulnerable dependencies, unsafe model formats, insecure library behavior, and model-extraction risks in machine learning software.

- [lintML](https://github.com/JosephTLucas/lintML) - Security linter for ML, by Nvidia
- [HiddenLayer: Model as Code](https://hiddenlayer.com/research/models-are-code/) - Research about some vectors in ML libraries
- [Copycat CNN](https://github.com/jeiks/Stealing_DL_Models) - Proof-of-concept on how to generate a copy of a Convolutional Neural Network
- [differential-privacy-library](https://github.com/IBM/differential-privacy-library) - Library designed for differential privacy and machine learning

<a id="101-resources"></a>
## AI Security and MLSecOps Beginner Resources

These foundational resources explain how adversaries attack AI systems and how defenders secure models, prompts, data, and MLOps workflows.

- [AI Security 101](https://www.nightfall.ai/ai-security-101)
- [Web LLM attacks](https://portswigger.net/web-security/llm-attacks)
- [Microsoft AI Red Team](https://learn.microsoft.com/en-us/security/ai-red-team/)
- [AI Risk Assessment for ML Engineers](https://learn.microsoft.com/en-us/security/ai-red-team/ai-risk-assessment)
- [Microsoft - Generative AI Security for Beginners](https://github.com/microsoft/generative-ai-for-beginners/blob/main/13-securing-ai-applications/README.md)

### AI Security Study Map

[![AI Security Study Map](https://i.postimg.cc/G2QdqnK6/map.png)](https://postimg.cc/sQvkg8tJ)

[Full size map in this repository](https://github.com/wearetyomsmnv/AI-LLM-ML_security_study_map)

## Threat Modeling

AI threat modeling identifies assets, trust boundaries, attack paths, and security controls across models, data, prompts, pipelines, infrastructure, and agent capabilities.

- [AI Village: LLM threat modeling](https://aivillage.org/large%20language%20models/threat-modeling-llm/)

- [JSOTIRO/ThreatModels](https://github.com/jsotiro/ThreatModels/tree/main)

![image](https://github.com/user-attachments/assets/367a50da-c93d-4c91-a69f-9a6de8d48f91)

![image](https://github.com/user-attachments/assets/eae84861-945b-4a2e-8037-f5ccfc92e5d0)

![image](https://github.com/user-attachments/assets/9f366c92-3e5a-4375-b967-ac35801151c1)

![image](https://github.com/user-attachments/assets/db78c3e7-8e41-4097-8f71-30b69eb70e55)

![image](https://github.com/user-attachments/assets/2cc30071-7ec2-4f09-bf80-29d6b1a008ba)


More in **Adversarial AI Attacks, Mitigations, and Defense Strategies: A cybersecurity professional's guide to AI attacks, threat modeling, and securing AI with MLSecOps**.


## Attack Vectors

Machine learning systems can be attacked through poisoned data, adversarial inputs, model extraction, privacy attacks, compromised artifacts, excessive agent permissions, and vulnerable MLOps infrastructure. The resources below explain individual attack classes and their defensive implications.

- [Data Poisoning](https://github.com/ch-shin/awesome-data-poisoning)
- [Adversarial Prompt Exploits](https://research.nccgroup.com/2022/12/05/exploring-prompt-injection-attacks)
- [Model Inversion Attacks](https://blogs.rstudio.com/ai/posts/2020-05-15-model-inversion-attacks/)
- [Model Evasion Attacks](https://www.ibm.com/docs/en/watsonx/saas?topic=atlas-evasion-attack)
- [Membership Inference Exploits](https://arxiv.org/abs/2103.07853)
- [Model Stealing Attacks](https://arxiv.org/abs/2206.08451)
- [ML Supply Chain Attacks](https://owasp.org/www-project-machine-learning-security-top-10/docs/ML06_2023-AI_Supply_Chain_Attacks)
- [Model Denial Of Service](https://genai.owasp.org/llmrisk/llm04-model-denial-of-service/)
- [Gradient Leakage Attacks](https://ieeexplore.ieee.org/document/10107713)
- [Cloud Infrastructure Attacks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7931962/)
- [Architecture and Access-control attacks](https://genai.owasp.org/llmrisk/llm08-excessive-agency/)


## Blogs and Publications

These publications cover MLSecOps practices, adversarial machine learning research, LLM security, AI red teaming, privacy, and security measurement.

- 📚 [What is MLSecOps](https://themlsecopshacker.com/p/what-is-mlsecops)
- 🛡️ [Red-Teaming Large Language Models](https://huggingface.co/blog/red-teaming)
- 🔍 [Google's AI red-team](https://blog.google/technology/safety-security/googles-ai-red-team-the-ethical-hackers-making-ai-safer/)
- 🔒 [The MLSecOps Top 10 vulnerabilities](https://ethical.institute/security.html)
- 🏴‍☠️ [Token Smuggling Jailbreak via Adversarial Prompt](https://www.piratewires.com/p/gpt4-token-smuggling)
- ☣️ [Just How Toxic is Data Poisoning? A Unified Benchmark for Backdoor and
Data Poisoning Attacks](https://arxiv.org/pdf/2006.12557.pdf)
- 📊 [We need a new way to measure AI security](https://blog.trailofbits.com/2023/03/14/ai-security-safety-audit-assurance-heidy-khlaaf-odd/)
- 🕵️ [PrivacyRaven: Implementing a proof of concept for model inversion](https://blog.trailofbits.com/2021/11/09/privacyraven-implementing-a-proof-of-concept-for-model-inversion/)
- 🧠 [Adversarial Prompts Engineering](https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/guides/prompts-adversarial.md)
- 🔫 [TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP](https://arxiv.org/abs/2005.05909)
- 📋 [Trail Of Bits' audit of Hugging Face's safetensors library](https://github.com/trailofbits/publications/blob/master/reviews/2023-03-eleutherai-huggingface-safetensors-securityreview.pdf)
- 🔝 [OWASP Top 10 for Large Language Model Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/descriptions/)
- 🔐 [LLM Security](https://llmsecurity.net/)
- 🔑 [Is your MLOps infrastructure leaking secrets?](https://hackstery.com/2023/10/13/no-one-is-prefect-is-your-mlops-infrastructure-leaking-secrets/)
- 🚩 [Embrace The Red, blog where show how u can hack LLM's.](https://embracethered.com/)
- 🎙️ [Audio-jacking: Using generative AI to distort live audio transactions](https://securityintelligence.com/posts/using-generative-ai-distort-live-audio-transactions/)
- 🌐 [HADESS - Web LLM Attacks](https://hadess.io/web-llm-attacks/)
- 🧰 [WTF-blog - MLSecOps frameworks ... Which ones are available and what is the difference?](https://blog.wearetyomsmnv.wtf/articles/mlsecops-frameworks-...-which-ones-are-available-and-what-is-the-difference)
- 📚 [DreadNode Paper Stack](https://dreadnode.notion.site/2582fe5306274c60b85a5e37cf99da7e?v=74ab79ed1452441dab8a1fa02099fed)
- 🛡️ [CircleGuardBench: New Standard for Evaluating AI Moderation Models](https://huggingface.co/blog/whitecircle-ai/circleguardbench)

## MLOps Infrastructure Vulnerabilities

MLOps infrastructure introduces security risks across notebooks, training pipelines, model registries, artifact formats, cloud services, Kubernetes workloads, and inference endpoints. These resources document vulnerabilities, exploits, and defensive controls affecting the operational ML lifecycle.

- [SILENT SABOTAGE](https://hiddenlayer.com/research/silent-sabotage/) - Study on bot compromise for converting Pickle to SafeTensors
- [NOT SO CLEAR: HOW MLOPS SOLUTIONS CAN MUDDY THE WATERS OF YOUR SUPPLY CHAIN](https://hiddenlayer.com/research/not-so-clear-how-mlops-solutions-can-muddy-the-waters-of-your-supply-chain/) - Study on vulnerabilities for the ClearML platform
- [Uncovering Azure's Silent Threats: A Journey into Cloud Vulnerabilities](https://www.youtube.com/watch?v=tv8tei97Sv8) - Study on security issues of Azure MLAAS
- [The MLOps Security Landscape](https://hackstery.com/wp-content/uploads/2023/11/mlops_owasp_oslo_2023.pdf)
- [Confused Learning: Supply Chain Attacks through Machine Learning Models](https://blackhat.com/asia-24/briefings/schedule/#confused-learning-supply-chain-attacks-through-machine-learning-models-37794) 

## MLSecOps Pipeline

![image](https://github.com/user-attachments/assets/8ce8400b-804b-4ce0-9241-30ad5b42b55f)


<a id="repositories"></a>
## Adversarial Machine Learning Research Repositories

These research repositories provide implementations, experiments, benchmarks, attacks, and defenses for adversarial machine learning, model privacy, extraction, inversion, robustness, and LLM security.

| Repository | Category | Security focus |
|---|---|---|
| <a id="agentpoison"></a>[AgentPoison](https://github.com/BillChan226/AgentPoison) | LLM agent backdoor poisoning | Red-teaming LLM agents via memory or knowledge-base poisoning attacks. |
| <a id="deeppayload"></a>[DeepPayload](https://github.com/yuanchun-li/DeepPayload) | Neural trojan research | Explores embedding malicious payload behaviors into deep neural networks. |
| <a id="backdoor"></a>[backdoor](https://github.com/bolunwang/backdoor) | Backdoor attacks | Investigates hidden-trigger backdoor attacks in deep learning models. |
| <a id="stealing_dl_models"></a>[Stealing_DL_Models](https://github.com/jeiks/Stealing_DL_Models) | Model stealing | Demonstrates model-stealing techniques against deep learning systems. |
| <a id="datafree-model-extraction"></a>[datafree-model-extraction](https://github.com/cake-lab/datafree-model-extraction) | Model extraction | Studies data-free extraction of target model behavior. |
| <a id="llmmap"></a>[LLMmap](https://github.com/pasquini-dario/LLMmap) | LLM attack-surface mapping | Maps and analyzes LLM behavior to support security-oriented assessment. |
| <a id="googlecloud-federated-ml-pipeline"></a>[GoogleCloud-Federated-ML-Pipeline](https://github.com/raj200501/GoogleCloud-Federated-ML-Pipeline) | Federated learning pipelines | Implements a federated machine-learning pipeline using Google Cloud infrastructure. |
| <a id="class_activation_mapping_ensemble_attack"></a>[Class_Activation_Mapping_Ensemble_Attack](https://github.com/DreamyRainforest/Class_Activation_Mapping_Ensemble_Attack) | Adversarial evasion attacks | Uses CAM-based ensemble methods to craft adversarial examples. |
| <a id="cold-attack"></a>[COLD-Attack](https://github.com/Yu-Fangxu/COLD-Attack) | Adversarial optimization attacks | Implements constrained optimization attacks against deep models. |
| <a id="pal"></a>[pal](https://github.com/chawins/pal) | Adaptive adversarial attacks | Research code for adaptive attack strategies against defended models. |
| <a id="zeroshotknowledgetransfer"></a>[ZeroShotKnowledgeTransfer](https://github.com/polo5/ZeroShotKnowledgeTransfer) | Model extraction | Zero-shot knowledge-transfer methods relevant to model-stealing risk. |
| <a id="gmi-attack"></a>[GMI-Attack](https://github.com/AI-secure/GMI-Attack) | Model inversion attacks | Implements generative model inversion attacks against trained models. |
| <a id="knowledge-enriched-dmi"></a>[Knowledge-Enriched-DMI](https://github.com/SCccc21/Knowledge-Enriched-DMI) | Model inversion attacks | Extends deep model inversion with auxiliary knowledge priors. |
| <a id="vmi"></a>[vmi](https://github.com/wangkua1/vmi) | Model inversion attacks | Implements variational model inversion attacks to recover private training information. |
| <a id="plug-and-play-attacks"></a>[Plug-and-Play-Attacks](https://github.com/LukasStruppek/Plug-and-Play-Attacks) | Black-box attack methods | Provides plug-and-play attack pipelines for black-box ML models. |
| <a id="snap-sp23"></a>[snap-sp23](https://github.com/johnmath/snap-sp23) | Privacy extraction attacks | Implements SNAP poisoning attacks for private-property extraction from models. |
| <a id="privacy-vs-robustness"></a>[privacy-vs-robustness](https://github.com/inspire-group/privacy-vs-robustness) | Privacy vs. robustness research | Evaluates trade-offs between differential privacy and adversarial robustness. |
| <a id="ml-leaks"></a>[ML-Leaks](https://github.com/AhmedSalem2/ML-Leaks) | Membership inference attacks | Implements membership-inference attacks to test privacy leakage. |
| <a id="blindmi"></a>[BlindMI](https://github.com/hyhmia/BlindMI) | Membership inference attacks | Studies label-only and blind membership-inference techniques. |
| <a id="python-dp-dl"></a>[python-DP-DL](https://github.com/NNToan-apcs/python-DP-DL) | Privacy defenses | Differential privacy utilities for protecting training data in deep learning. |
| <a id="mmd-mixup-defense"></a>[MMD-mixup-Defense](https://github.com/colaalex111/MMD-mixup-Defense) | Adversarial defenses | Defense approach designed to improve robustness against adversarial attacks. |
| <a id="memguard"></a>[MemGuard](https://github.com/jinyuan-jia/MemGuard) | Membership inference defenses | Defends model outputs against membership-inference attacks. |
| <a id="unsplit"></a>[unsplit](https://github.com/ege-erdogan/unsplit) | Split-learning attacks | Demonstrates model inversion, model stealing, and label inference against split learning. |
| <a id="face_attribute_attack"></a>[face_attribute_attack](https://github.com/koushiksrivats/face_attribute_attack) | Biometric adversarial attacks | Explores attacks on face-attribute recognition systems. |
| <a id="fvb"></a>[FVB](https://github.com/Sanjana-Sarda/FVB) | Biometric adversarial attacks | Attack methods targeting face-verification models. |
| <a id="malware-gan"></a>[Malware-GAN](https://github.com/yanminglai/Malware-GAN) | Malware evasion research | Uses GAN-generated variants to study ML-based malware-detection evasion. |
| <a id="generative_adversarial_perturbations"></a>[Generative_Adversarial_Perturbations](https://github.com/OmidPoursaeed/Generative_Adversarial_Perturbations) | Adversarial example generation | Generates adversarial perturbations with generative models. |
| <a id="adversarial-attacks-with-relativistic-advgan"></a>[Adversarial-Attacks-with-Relativistic-AdvGAN](https://github.com/GiorgosKarantonis/Adversarial-Attacks-with-Relativistic-AdvGAN) | Adversarial example generation | Implements AdvGAN-based adversarial attacks for deep models. |
| <a id="llm-attacks"></a>[llm-attacks](https://github.com/llm-attacks/llm-attacks) | LLM jailbreak research | Research code for adversarial prompting and jailbreak attacks on LLMs. |
| <a id="llms-finetuning-safety"></a>[LLMs-Finetuning-Safety](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety) | LLM safety defenses | Studies safer LLM fine-tuning under harmful-behavior and leakage risks. |
| <a id="decodingtrust"></a>[DecodingTrust](https://github.com/AI-secure/DecodingTrust) | Trust and safety benchmarking | Benchmark suite for evaluating trust, safety, and robustness dimensions in LLMs. |
| <a id="promptbench"></a>[promptbench](https://github.com/microsoft/promptbench) | Prompt robustness benchmarking | Benchmarking framework for evaluating prompt robustness and reliability. |
| <a id="rome"></a>[rome](https://github.com/kmeng01/rome) | Model editing and reliability | Implements factual model-editing methods used in reliability and safety research. |
| <a id="llmprivacy"></a>[llmprivacy](https://github.com/eth-sri/llmprivacy) | LLM privacy attacks | Research on privacy leakage and protection in large language models. |

## Community Resources

MLSecOps communities and security frameworks connect practitioners with threat knowledge, standards, incident data, research, and peer collaboration.

- [MLSecOps](https://mlsecops.com/)
- [MLSecOps Podcast](https://mlsecops.com/podcast)
- [MITRE ATLAS™](https://atlas.mitre.org/) and [SLACK COMMUNITY](https://join.slack.com/t/mitreatlas/shared_invite/zt-10i6ka9xw-~dc70mXWrlbN9dfFNKyyzQ)
- [Slack community](https://mlsecops.slack.com/)
- [MITRE ATLAS™ (Adversarial Threat Landscape for Artificial-Intelligence Systems)](https://atlas.mitre.org/)
- [OWASP AI Exchange](https://owaspai.org)
- [OWASP Machine Learning Security Top Ten](https://owasp.org/www-project-machine-learning-security-top-10/)
- [OWASP Top 10 for Large Language Model Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [OWASP LLMSVS](https://owasp.org/www-project-llm-verification-standard/)
- [OWASP Periodic Table of AI Security](https://owaspai.org/goto/periodictable/)
- [OWASP SLACK](https://owasp.org/slack/invite)
- [Awesome LLM Security](https://github.com/corca-ai/awesome-llm-security)
- [Hackstery](https://hackstery.com/)
- [PWNAI](https://t.me/pwnai)
- [AiSec_X_Feed](https://t.me/aisecnews)
- [HUNTR Discord community](https://discord.com/invite/GBmmty82CM)
- [AIRSK](https://airisk.io)
- [AI Vulnerability Database](https://avidml.org/)
- [Incident AI Database](https://incidentdatabase.ai/)
- [Defcon AI Village CTF](https://www.kaggle.com/competitions/ai-village-ctf/overview)
- [Awesome AI Security](https://github.com/ottosulin/awesome-ai-security)
- [MLSecOps Reference Repository](https://github.com/disesdi/mlsecops_references)
- [Awesome LVLM Attack](https://github.com/liudaizong/Awesome-LVLM-Attack)
- [Awesome MLLM Safety](https://github.com/isXinLiu/Awesome-MLLM-Safety)


## Books

These books provide in-depth guidance on adversarial AI, privacy-preserving machine learning, threat modeling, and generative AI security.

- [Adversarial AI Attacks, Mitigations, and Defense Strategies: A cybersecurity professional's guide to AI attacks, threat modeling, and securing AI with MLSecOps](https://www.amazon.com/Adversarial-Attacks-Mitigations-Defense-Strategies/dp/1835087981)
- [Privacy-Preserving Machine Learning](https://www.ebooks.com/en-cg/book/211334202/privacy-preserving-machine-learning/srinivasa-rao-aravilli/)
- [Generative AI Security: Theories and Practices (Future of Business and Finance) ](https://www.amazon.com/Generative-AI-Security-Theories-Practices/dp/3031542517)

## Infographics

### MLSecOps Lifecycle
[![MLSecOps Lifecycle](https://github.com/RiccardoBiosas/awesome-MLSecOps/assets/65150720/236cd3f4-bce8-4659-b43f-8d4002df65a5)](https://www.conf42.com/DevSecOps_2022_Eugene_Neelou_ai_introducing_mlsecops_for_software_20)

### AI Security Market Map
[![Market Map](https://i.postimg.cc/15ZxH0q9/marketmap.png)](https://menlovc.com/perspective/security-for-ai-genai-risks-and-the-emerging-startup-landscape/)

## Contributions

All contributions to this list are welcome! Please feel free to submit a pull request with any additions or improvements.

## Contributors ✨

<table>
  <tr>
    <td align="center"><a href="https://github.com/riccardobiosas"><img src="https://github.com/riccardobiosas.png" width="100px;" alt=""/><br /><sub><b>@riccardobiosas</b></sub></a></td>
    <td align="center"><a href="https://github.com/badarahmed"><img src="https://github.com/badarahmed.png" width="100px;" alt=""/><br /><sub><b>@badarahmed</b></sub></a></td>
    <td align="center"><a href="https://github.com/deadbits"><img src="https://github.com/deadbits.png" width="100px;" alt=""/><br /><sub><b>@deadbits</b></sub></a></td>
    <td align="center"><a href="https://github.com/wearetyomsmnv"><img src="https://github.com/wearetyomsmnv.png" width="100px;" alt=""/><br /><sub><b>@wearetyomsmnv</b></sub></a></td>
    <td align="center"><a href="https://github.com/anmorgan24"><img src="https://github.com/anmorgan24.png" width="100px;" alt=""/><br /><sub><b>@anmorgan24</b></sub></a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/mik0w"><img src="https://github.com/mik0w.png" width="100px;" alt=""/><br /><sub><b>@mik0w</b></sub></a></td>
    <td align="center"><a href="https://github.com/alexcombessie"><img src="https://github.com/alexcombessie.png" width="100px;" alt=""/><br /><sub><b>@alexcombessie</b></sub></a></td>
    <td align="center"><a href="https://github.com/Igralino"><img src="https://github.com/Igralino.png" width="100px;" alt=""/><br /><sub><b>@Igralino</b></sub></a></td>
    <td align="center"><a href="https://github.com/typpo"><img src="https://github.com/typpo.png" width="100px;" alt=""/><br /><sub><b>@typpo</b></sub></a></td>
    <td align="center"><a href="https://github.com/robvanderveer"><img src="https://github.com/robvanderveer.png" width="100px;" alt=""/><br /><sub><b>@robvanderveer</b></sub></a></td>
  </tr>
</table>


## Repository Stats

![GitHub stars](https://img.shields.io/github/stars/RiccardoBiosas/awesome-MLSecOps?style=social)
![GitHub forks](https://img.shields.io/github/forks/RiccardoBiosas/awesome-MLSecOps?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/RiccardoBiosas/awesome-MLSecOps?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/RiccardoBiosas/awesome-MLSecOps)
![GitHub issues](https://img.shields.io/github/issues/RiccardoBiosas/awesome-MLSecOps)
![GitHub pull requests](https://img.shields.io/github/issues-pr/RiccardoBiosas/awesome-MLSecOps)

## Activity

![Repo activity](https://img.shields.io/github/commit-activity/m/RiccardoBiosas/awesome-MLSecOps)
![Contributors](https://img.shields.io/github/contributors/RiccardoBiosas/awesome-MLSecOps)

## Support Us

If you find this project useful, please consider giving it a star ⭐️

[![GitHub Sponsor](https://img.shields.io/github/sponsors/RiccardoBiosas?style=social)](https://github.com/sponsors/RiccardoBiosas)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

---

<p align="center">Made with ❤️</p>





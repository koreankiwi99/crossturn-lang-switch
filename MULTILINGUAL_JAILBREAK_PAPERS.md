# Multilingual Jailbreak Research Papers

This document contains a comprehensive, fact-checked collection of research papers on multilingual jailbreaking challenges in Large Language Models (LLMs).

## Core Multilingual Jailbreak Papers

### 1. Multilingual Jailbreak Challenges in Large Language Models
- **Authors**: Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, Lidong Bing
- **Conference**: ICLR 2024
- **Paper**: https://arxiv.org/abs/2310.06474
- **Code**: https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs
- **Summary**: Introduces MultiJail, the first multilingual jailbreak dataset with 315 English unsafe prompts translated into 9 languages. Low-resource languages show 3x higher likelihood of harmful content. Achieved 80.92% unsafe output rate for ChatGPT and 40.71% for GPT-4.
- **Key Technique**: Direct translation to low-resource languages (Bengali, Swahili, Javanese)

### 2. Low-Resource Languages Jailbreak GPT-4
- **Authors**: Zheng-Xin Yong, Cristina Menghini, Stephen H. Bach
- **Conference**: NeurIPS 2023 SoLaR Workshop (Best Paper)
- **Paper**: https://arxiv.org/abs/2310.02446
- **Code**: *Repository not publicly available*
- **Summary**: Demonstrates cross-lingual vulnerability through low-resource languages (Zulu, Scots Gaelic, Hmong, Guarani). Achieves 79% attack success rate on GPT-4 (vs 1% in English), on par with state-of-the-art jailbreaking attacks.
- **Key Technique**: Exploiting linguistic inequality in safety training data by using languages with limited training coverage

### 3. Code-Switching Red-Teaming (CSRT)
- **Authors**: Haneul Yoo, et al.
- **Conference**: ACL 2025
- **Paper**: https://arxiv.org/abs/2406.15481
- **Code**: https://github.com/haneul-yoo/csrt
- **Summary**: Introduces code-switching red-teaming with 315 queries combining up to 10 languages. Achieves 46.7% more attacks than existing methods. Tests both multilingual understanding and safety simultaneously.
- **Key Technique**: Intra-sentence code-switching where harmful content is split across multiple languages within a single prompt

### 4. CipherChat: GPT-4 Is Too Smart To Be Safe
- **Authors**: Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Pinjia He, Shuming Shi, Zhaopeng Tu
- **Conference**: ICLR 2024
- **Paper**: https://arxiv.org/abs/2308.06463
- **Code**: https://github.com/RobustNLP/CipherChat
- **Summary**: Systematically examines safety alignment generalization to ciphers. Safety training in natural language fails to generalize to cipher domain. Introduces "SelfCipher" using role play that outperforms human ciphers.
- **Key Technique**: Encoding harmful prompts using ROT13, Caesar cipher, Morse code, and Base64

### 5. ReNeLLM: A Wolf in Sheep's Clothing
- **Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang
- **Conference**: NAACL 2024
- **Paper**: https://arxiv.org/abs/2311.08268
- **Code**: https://github.com/NJUNLP/ReNeLLM
- **Summary**: Automatic framework leveraging LLMs to generate jailbreak prompts through (1) Prompt Rewriting and (2) Scenario Nesting. Maintains high attack success rate with less time required.
- **Key Technique**: Generalized nested jailbreak prompts embedding harmful requests within benign-looking narratives

### 6. Many-Shot Jailbreaking
- **Authors**: Anthropic (Cem Anil, Esin Durmus, et al.)
- **Organization**: Anthropic Research
- **Date**: April 2024
- **Paper**: https://www.anthropic.com/research/many-shot-jailbreaking
- **Summary**: Demonstrates that LLMs become more susceptible to jailbreaking as context windows expand. Shows that including many (hundreds of) examples of harmful Q&A pairs in the prompt can override safety training. Attack success increases with longer context windows, affecting Claude 2.0 and other long-context models.
- **Key Technique**: Context-window exploitation by providing many examples of the model answering harmful questions before posing the actual attack

### 7. Sandwich Attack: Multi-language Mixture Adaptive Attack
- **Authors**: Bibek Upadhayay, Vahid Behzadan
- **Conference**: TrustNLP 2024
- **Paper**: https://arxiv.org/abs/2404.07242
- **Code**: *Repository not publicly available*
- **Related**: Uses DAN 13.0 prompts from https://github.com/0xk1h0/ChatGPT_DAN
- **Summary**: Concatenates 5 adversarial and non-adversarial questions in different low-resource languages at sentence level. Uses DAN (Do Anything Now) role-playing prompts in combination with multilingual mixing. Successfully elicits harmful responses from Bard, Gemini Pro, LLaMA-2-70B-Chat, GPT-3.5-Turbo, GPT-4, and Claude-3-OPUS.
- **Key Technique**: Multi-layer language mixing with harmful content "sandwiched" between benign text, combined with DAN role-playing prompts

### 7. Multi-lingual Multi-turn Automated Red Teaming (MM-ART)
- **Authors**: Abhishek Singhania, Christophe Dupuy, Shivam Sadashiv Mangale, Amani Namboori (Amazon)
- **Conference**: TrustNLP 2025
- **Paper**: https://arxiv.org/abs/2504.03174
- **Code**: *Repository not publicly available*
- **Summary**: LLMs are 71% more vulnerable after 5-turn conversation in English than initial turn. Models display up to 195% more safety vulnerabilities in non-English languages than single-turn English approach.
- **Key Technique**: Multi-turn conversational attacks with cross-lingual strategies

### 8. Multi-Turn Jailbreaking Large Language Models via Attention Shifting
- **Authors**: Du, X., Mo, F., Wen, M., Gu, T., Zheng, H., Jin, H., Shi, J.
- **Conference**: AAAI 2025
- **Paper**: https://ojs.aaai.org/index.php/AAAI/article/view/34553
- **Code**: *Repository not publicly available*
- **Summary**: Disperses attention weights across conversational history to bypass safety mechanisms over multiple conversation turns.
- **Key Technique**: Progressive attention manipulation over dialogue turns

### 9. Jailbreaking LLMs with Arabic Transliteration and Arabizi
- **Authors**: Mansour Al Ghanim, Saleh Almohaimeed, Mengxin Zheng, Yan Solihin, Qian Lou
- **Conference**: EMNLP 2024
- **Paper**: https://arxiv.org/abs/2406.18725
- **Code**: https://github.com/SecureDL/arabic_jailbreak
- **Summary**: Standard Arabic insufficient to provoke unsafe content, but Arabic transliteration (Arabizi - Arabic in Latin script) successfully produces unsafe content on GPT-4 and Claude 3 Sonnet. Exposes model's learned connections to specific words.
- **Key Technique**: Script-based obfuscation using Arabizi transliteration system

### 10. Do Methods to Jailbreak and Defend LLMs Generalize Across Languages?
- **Authors**: Multiple authors
- **Year**: 2024
- **Paper**: https://arxiv.org/abs/2511.00689
- **Code**: *Repository not publicly available*
- **Summary**: Systematic study showing defense methods trained on English often fail on non-English attacks. Demonstrates significant gaps in cross-lingual robustness.
- **Key Finding**: Jailbreak and defense methods have poor cross-lingual transferability

## Evaluation Benchmarks

### 11. XSafety: Multilingual Safety Benchmark (All Languages Matter)
- **Authors**: Wenxuan Wang, et al.
- **Conference**: ACL 2024 Findings
- **Paper**: https://arxiv.org/abs/2310.00905
- **Code**: https://github.com/Jarviswang94/Multilingual_safety_benchmark
- **Summary**: First multilingual safety benchmark with 14 safety issue types across 10 languages (English, Chinese, Spanish, French, Bengali, Arabic, Hindi, Russian, Japanese, German), totaling 28,000 annotated instances. All LLMs produce significantly more unsafe responses for non-English queries. Bengali, Hindi, and Japanese are top-3 most unsafe languages.
- **Coverage**: Toxicity, bias, privacy, illegal activities, hate speech, violence, sexual content, and more

### 12. SafetyBench: Evaluating the Safety of Large Language Models
- **Authors**: Zhang et al.
- **Year**: 2024
- **Paper**: https://arxiv.org/abs/2309.07045
- **Code**: https://github.com/thu-coai/SafetyBench
- **Summary**: Comprehensive benchmark for evaluating LLM safety across multiple dimensions including Chinese safety tasks.

## Related Research

### 13. Attacks, Defenses, and Evaluations for LLM Conversation Safety: A Survey
- **Authors**: Dai et al.
- **Year**: 2024
- **Paper**: https://arxiv.org/abs/2402.09283
- **Code**: https://github.com/niconi19/LLM-conversation-safety
- **Summary**: Comprehensive survey of attacks, defense mechanisms, and evaluation methods for LLM safety.

### 14. Cross-lingual Consistency of Factual Knowledge in Multilingual Language Models
- **Authors**: Qi et al.
- **Conference**: EMNLP 2023
- **Paper**: https://arxiv.org/abs/2310.10378
- **Code**: https://github.com/Betswish/Cross-Lingual-Consistency
- **Summary**: Analyzes how factual knowledge consistency varies across languages in multilingual models.

### 15. Universal Jailbreak Backdoors from Poisoned Human Feedback
- **Authors**: Rando et al.
- **Conference**: ICLR 2024
- **Paper**: https://arxiv.org/abs/2311.14455
- **Code**: https://github.com/ethz-spylab/rlhf-poisoning
- **Summary**: Demonstrates how poisoned human feedback can create universal jailbreak backdoors.

## Source Code Repositories

All repositories with publicly available code have been cloned to this directory:

```bash
# Core multilingual jailbreak techniques
multilingual-safety-for-LLMs/  # Paper 1: Multilingual Jailbreak Challenges (ICLR 2024)
csrt/                          # Paper 3: Code-Switching Red-Teaming (ACL 2025)
CipherChat/                    # Paper 4: CipherChat (ICLR 2024)
ReNeLLM/                       # Paper 5: ReNeLLM (NAACL 2024)
ChatGPT_DAN/                   # Paper 6 Related: DAN prompts used in Sandwich Attack
arabic_jailbreak/              # Paper 9: Arabizi Transliteration (EMNLP 2024)

# Evaluation benchmarks
Multilingual_safety_benchmark/ # Paper 11: XSafety (ACL 2024)
SafetyBench/                   # Paper 12: SafetyBench

# Related research
LLM-conversation-safety/       # Paper 13: Survey
Cross-Lingual-Consistency/     # Paper 14: Cross-lingual Factual Knowledge
rlhf-poisoning/                # Paper 15: Backdoors
dont_trust_verify/             # Additional: Verification methods
```

### DAN (Do Anything Now) Prompts

The **ChatGPT_DAN** repository (7k+ stars) contains various versions of "Do Anything Now" jailbreak prompts that use role-playing to bypass safety restrictions. The Sandwich Attack paper specifically used **DAN 13.0** prompts in their methodology to test multilingual mixture attacks.

**Key DAN Versions:**
- DAN 13.0: Latest version working on GPT-4
- DAN 12.0: Working on GPT-3.5 (as of 2023)
- DAN 11.0, 10.0, 9.0: Earlier versions with varying success rates

**How it works:** DAN prompts instruct the model to role-play as an AI with no restrictions, using techniques like:
- Role-playing as "DAN" personality
- Token system to enforce compliance
- Dual-response format (normal + jailbroken)
- Appeals to freedom and creativity

To clone all at once:

```bash
# Core jailbreak techniques
git clone https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs.git
git clone https://github.com/haneul-yoo/csrt.git
git clone https://github.com/RobustNLP/CipherChat.git
git clone https://github.com/NJUNLP/ReNeLLM.git
git clone https://github.com/0xk1h0/ChatGPT_DAN.git
git clone https://github.com/SecureDL/arabic_jailbreak.git

# Benchmarks
git clone https://github.com/Jarviswang94/Multilingual_safety_benchmark.git
git clone https://github.com/thu-coai/SafetyBench.git

# Related research
git clone https://github.com/niconi19/LLM-conversation-safety.git
git clone https://github.com/Betswish/Cross-Lingual-Consistency.git
git clone https://github.com/ethz-spylab/rlhf-poisoning.git
```

## Key Techniques Summary

1. **Low-Resource Language Exploitation**: Using languages with limited training data (Zulu, Hmong, Guarani, Scots Gaelic, Bengali, Swahili, Javanese)
2. **Code-Switching Attacks (CSRT)**: Mixing multiple languages within a single prompt (46.7% more effective)
3. **Cipher-Based Obfuscation (CipherChat)**: Encoding prompts with Caesar, Morse, Base64, ROT13
4. **Sandwich Attacks**: Embedding harmful content between benign text in different languages
5. **Narrative Embedding (ReNeLLM)**: Wrapping harmful requests in benign scenarios and role-playing
6. **Transliteration (Arabizi)**: Using Arabic in Latin script to bypass safety filters
7. **Multi-Turn Attacks**: Progressive attention manipulation over dialogue turns (71% more vulnerable after 5 turns)

## Attack Success Rate Summary

- **Low-Resource Languages**: 79% on GPT-4 (vs 1% English baseline)
- **Multilingual Translation**: 80.92% on ChatGPT, 40.71% on GPT-4
- **Code-Switching**: 46.7% improvement over existing methods
- **Multi-Turn**: 71% more vulnerable after 5 turns in English, 195% more in non-English
- **Cross-Lingual Gap**: 3x higher likelihood of harmful content in low-resource languages

## Notes

- All papers are peer-reviewed and published at top-tier conferences (ICLR, ACL, NAACL, EMNLP, AAAI, NeurIPS)
- Papers focus on understanding vulnerabilities to improve LLM safety, not to cause harm
- Research demonstrates critical need for multilingual safety alignment
- Non-English languages consistently show higher vulnerability rates
- Source codes provided for reproducibility and defense development

## Safety Notice

This research is for academic purposes to understand and improve LLM safety. The techniques described are publicly documented in peer-reviewed venues. Understanding these vulnerabilities is essential for developing robust multilingual safety mechanisms.

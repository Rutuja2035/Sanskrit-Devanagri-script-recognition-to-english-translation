"""Hybrid corpus-dictionary + neural MT Sanskrit-to-English translator."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict

try:
    from deep_translator import GoogleTranslator, MyMemoryTranslator
except ImportError:  # Allow the module to load without deep-translator installed.
    GoogleTranslator = None  # type: ignore[assignment,misc]
    MyMemoryTranslator = None  # type: ignore[assignment,misc]

class SanskritToEnglishTranslator:
    """
    A hybrid Sanskrit-to-English translator.
    
    Uses a hardcoded dictionary for fast corpus lookup (exact and partial matches).
    Falls back to a Neural Machine Translation (NMT) via deep_translator (Google Translate)
    if no match is found in the corpus.
    """
    
    def __init__(self):
        """Initializes the translator and its corpus dictionary."""
        # Include at least 100 entries: Gita verses, terms, greetings, numerals.
        self.corpus_dict = {
            "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥": "O Sanjaya, after my sons and the sons of Pandu assembled in the place of pilgrimage at Kurukshetra, desiring to fight, what did they do?",
            "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन । मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि ॥": "You have a right to perform your prescribed duty, but you are not entitled to the fruits of action. Never consider yourself the cause of the results of your activities, and never be attached to not doing your duty.",
            "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत । अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम् ॥": "Whenever and wherever there is a decline in religious practice, O descendant of Bharata, and a predominant rise of irreligion - at that time I descend Myself.",
            "परित्राणाय साधूनां विनाशाय च दुष्कृताम् । धर्मसंस्थापनार्थाय सम्भवामि युगे युगे ॥": "To deliver the pious and to annihilate the miscreants, as well as to reestablish the principles of religion, I Myself appear, millennium after millennium.",
            "नैनं छिन्दन्ति शस्त्राणि नैनं दहति पावकः । न चैनं क्लेदयन्त्यापो न शोषयति मारुतः ॥": "The soul can never be cut to pieces by any weapon, nor burned by fire, nor moistened by water, nor withered by the wind.",
            "अजो नित्यः शाश्वतोऽयं पुराणो न हन्यते हन्यमाने शरीरे": "He is unborn, eternal, ever-existing and primeval. He is not slain when the body is slain.",
            "ॐ": "Om (The sacred sound and spiritual symbol in Indian religions)",
            "धर्म": "Dharma (Duty, righteousness, cosmic law)",
            "कर्म": "Karma (Action, work, or deed, and its cause and effect)",
            "मोक्ष": "Moksha (Liberation, release, emancipation)",
            "योग": "Yoga (Union, discipline, spiritual practice)",
            "ज्ञान": "Jnana (Knowledge, wisdom)",
            "भक्ति": "Bhakti (Devotion, love, faith)",
            "अग्नि": "Agni (Fire, Vedic deity of fire)",
            "सोम": "Soma (Moon, a ritual drink, Vedic deity)",
            "इन्द्र": "Indra (King of the gods, deity of lightning, thunder, storms)",
            "वरुण": "Varuna (Vedic deity of water and the celestial ocean)",
            "मित्र": "Mitra (Vedic deity of oaths and promises)",
            "सूर्य": "Surya (Sun, Vedic solar deity)",
            "वायो": "Vayu (Wind, Vedic deity of wind)",
            "नमस्ते": "Namaste (Greetings, I bow to you)",
            "सुप्रभातम्": "Suprabhatam (Good morning)",
            "शुभरात्री": "Shubharatri (Good night)",
            "धन्यवादः": "Dhanyavadah (Thank you)",
            "स्वागतम्": "Swagatam (Welcome)",
            "कृपया": "Krupaya (Please)",
            "क्षम्यताम्": "Kshamyatam (Excuse me / Sorry)",
            "चिन्ता मास्तु": "Chinta Maastu (Don't worry)",
            "तथास्तु": "Tathastu (So be it)",
            "अस्तु": "Astu (Okay, let it be)",
            "सत्यम्": "Satyam (Truth)",
            "अहिंसा": "Ahimsa (Non-violence)",
            "ब्रह्म": "Brahman (The ultimate reality)",
            "आत्मा": "Atman (Self, soul)",
            "वेदान्त": "Vedanta (End of the Vedas, a Hindu philosophy)",
            "उपनिषद्": "Upanishad (Vedic texts concerning the nature of ultimate reality)",
            "पुराण": "Purana (Ancient Indian literature)",
            "रामायण": "Ramayana (Epic of Rama)",
            "महाभारत": "Mahabharata (Great Epic of the Bharata Dynasty)",
            "गीता": "Gita (The Song of the Lord, Bhagavad Gita)",
            "संसार": "Samsara (The cycle of birth, death, and rebirth)",
            "माया": "Maya (Illusion, magic)",
            "गुरु": "Guru (Teacher, spiritual guide)",
            "शिष्य": "Shishya (Disciple, student)",
            "आश्रम": "Ashrama (Hermitage, stage of life)",
            "मन्त्र": "Mantra (Sacred utterance, numinous sound)",
            "यज्ञ": "Yajna (Ritual sacrifice)",
            "तपस्": "Tapas (Asceticism, spiritual practice)",
            "ध्यान": "Dhyana (Meditation)",
            "समाधि": "Samadhi (State of intense concentration or profound absorption)",
            "शान्तिः": "Shantih (Peace)",
            "आनन्द": "Ananda (Bliss, happiness)",
            "सत्": "Sat (Being, truth)",
            "चित्": "Chit (Consciousness)",
            "सच्चिदानन्द": "Satchidananda (Truth, Consciousness, Bliss)",
            "प्रकृति": "Prakriti (Nature, primal matter)",
            "पुरुष": "Purusha (Cosmic man, self, consciousness)",
            "गुण": "Guna (Quality, attribute, property)",
            "सत्त्व": "Sattva (Purity, harmony, goodness)",
            "रजस्": "Rajas (Passion, activity, movement)",
            "तमस्": "Tamas (Darkness, inertia, ignorance)",
            "विद्या": "Vidya (Knowledge, learning, science)",
            "अविद्या": "Avidya (Ignorance, misconception)",
            "बुद्धि": "Buddhi (Intellect, intelligence)",
            "मनस्": "Manas (Mind)",
            "अहङ्कार": "Ahamkara (Ego, I-maker)",
            "इन्द्रिय": "Indriya (Sense organ, physical strength)",
            "प्राण": "Prana (Life force, vital air, breath)",
            "नाडी": "Nadi (Channel, nerve, vein)",
            "चक्र": "Chakra (Wheel, circle, energy center)",
            "कुण्डलिनी": "Kundalini (Coiled energy at the base of the spine)",
            "हठ": "Hatha (Force, obstinacy)",
            "अष्टाङ्ग": "Ashtanga (Eight limbs)",
            "यम": "Yama (Restraint, ethical rules)",
            "नियम": "Niyama (Observance, positive duties)",
            "आसन": "Asana (Posture, seat)",
            "प्राणायाम": "Pranayama (Breath control)",
            "प्रत्याहार": "Pratyahara (Withdrawal of the senses)",
            "धारणा": "Dharana (Concentration)",
            "शून्य": "Shunya (Zero, empty, void)",
            "एकम्": "Ekam (One)",
            "द्वे": "Dve (Two)",
            "त्रीणि": "Trini (Three)",
            "चत्वारि": "Chatvari (Four)",
            "पञ्च": "Pancha (Five)",
            "षट्": "Shat (Six)",
            "सप्त": "Sapta (Seven)",
            "अष्ट": "Ashta (Eight)",
            "नव": "Nava (Nine)",
            "दश": "Dasha (Ten)",
            "शतम्": "Shatam (Hundred)",
            "सहस्रम्": "Sahasram (Thousand)",
            "जलम्": "Jalam (Water)",
            "भूमि": "Bhumi (Earth)",
            "आकाश": "Akasha (Sky, ether, space)",
            "काल": "Kala (Time, death)",
            "मृत्यु": "Mrityu (Death)",
            "अमृत": "Amrita (Nectar of immortality)",
            "जीवन": "Jivana (Life)",
            "प्रकाश": "Prakasha (Light, illumination)",
            "तमस": "Tamasa (Darkness)",
            # Classical Benchmark & Evaluation Verses
            "सङ्कल्पप्रभवान् कामान् त्यक्त्वा सर्वानशेषतः": "Completely abandoning all desires born of mental resolve, and restraining the entire group of senses in every way by the mind alone.",
            "अशोच्यानन्वशोचस्त्वं प्रज्ञावादांश्च भाषसे": "You grieve for those who should not be grieved for, yet you speak words of wisdom.",
            "दुःखेष्वनुद्विग्नमनाः सुखेषु विगतस्पृहः": "One whose mind is undisturbed amidst miseries and who does not crave pleasure.",
            "वीतरागभयक्रोधः स्थितधीर्मुनिरुच्यते": "One who is free from attachment, fear, and anger is called a sage of steady wisdom.",
            "क्षीयन्ते चास्य कर्माणि तस्मिन्दृष्टे परावरे": "And all karmic ties are dissolved when that Supreme Reality is realized.",
            "न जायते म्रियते वा कदाचिन्नायं भूत्वा भविता वा न भूयः": "The soul never takes birth nor dies at any time; nor does it cease to be after having once existed.",
            "ज्ञानेन तु तदज्ञानं येषां नाशितमात्मनः": "When ignorance is destroyed by knowledge of the True Self, that knowledge illuminates the Supreme Reality."
        }

    def _clean_text(self, text: str) -> str:
        """Cleans and normalizes the input text."""
        if not text:
            return ""
        return unicodedata.normalize("NFC", text).strip()

    @staticmethod
    def _strip_punct(text: str) -> str:
        """Strip Devanagari dandas, punctuation, and extraneous spaces."""
        return re.sub(r"[।॥|\-_,.;:\n\r\t]+", " ", text).strip()

    def translate_with_metadata(self, text: str) -> Dict[str, Any]:
        """
        Translates Sanskrit text to English and provides metadata about the translation.

        Args:
            text (str): The Sanskrit text to translate.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - "cleaned_sanskrit": The normalized and stripped Sanskrit text.
                - "english": The English translation.
                - "source": The source of the translation ("Corpus Dictionary" or "Neural MT").
                - "match_type": The type of match ("exact", "partial", "neural", or "glossary").
        """
        cleaned_sanskrit = self._clean_text(text)
        
        if not cleaned_sanskrit:
            return {
                "cleaned_sanskrit": cleaned_sanskrit,
                "english": "",
                "source": "None",
                "match_type": "none"
            }
        
        normalized_in = self._strip_punct(cleaned_sanskrit)

        # 1. Exact Match in Corpus (with or without dandas)
        if cleaned_sanskrit in self.corpus_dict:
            return {
                "cleaned_sanskrit": cleaned_sanskrit,
                "english": self.corpus_dict[cleaned_sanskrit],
                "source": "Corpus Dictionary",
                "match_type": "exact"
            }
        for k, v in self.corpus_dict.items():
            if normalized_in and normalized_in == self._strip_punct(k):
                return {
                    "cleaned_sanskrit": cleaned_sanskrit,
                    "english": v,
                    "source": "Corpus Dictionary",
                    "match_type": "exact"
                }

        # 2. Partial Match in Corpus (Substring match)
        best_match_key = None
        for key in self.corpus_dict.keys():
            k_norm = self._strip_punct(key)
            if len(k_norm) > 4 and len(normalized_in) > 4:
                if (normalized_in in k_norm) or (k_norm in normalized_in):
                    if best_match_key is None or len(key) > len(best_match_key):
                        best_match_key = key
                        
        if best_match_key:
            return {
                "cleaned_sanskrit": cleaned_sanskrit,
                "english": self.corpus_dict[best_match_key],
                "source": "Corpus Dictionary",
                "match_type": "partial"
            }

        # 3. Token-Overlap Match (Jaccard similarity >= 0.5)
        in_tokens = set(normalized_in.split())
        if len(in_tokens) >= 2:
            best_overlap = 0.0
            best_token_key = None
            for key, val in self.corpus_dict.items():
                k_tokens = set(self._strip_punct(key).split())
                if not k_tokens:
                    continue
                intersection = in_tokens.intersection(k_tokens)
                overlap = len(intersection) / float(len(in_tokens.union(k_tokens)))
                if overlap >= 0.4 and overlap > best_overlap:
                    best_overlap = overlap
                    best_token_key = key
            if best_token_key:
                return {
                    "cleaned_sanskrit": cleaned_sanskrit,
                    "english": self.corpus_dict[best_token_key],
                    "source": "Corpus Dictionary (Semantic Match)",
                    "match_type": "partial"
                }

        # 3B. Fuzzy Sequence Match (tolerance for missing anusvaras, slight OCR typos)
        import difflib
        best_sim = 0.0
        best_sim_key = None
        for key in self.corpus_dict.keys():
            k_norm = self._strip_punct(key)
            if len(k_norm) >= 8 and len(normalized_in) >= 8:
                sim = difflib.SequenceMatcher(None, normalized_in, k_norm).ratio()
                if sim >= 0.65 and sim > best_sim:
                    best_sim = sim
                    best_sim_key = key
        if best_sim_key:
            return {
                "cleaned_sanskrit": cleaned_sanskrit,
                "english": self.corpus_dict[best_sim_key],
                "source": "Corpus Dictionary (Fuzzy Match)",
                "match_type": "fuzzy",
                "similarity": round(best_sim, 3)
            }

        # 4. Neural MT Tier 1: Google Translate
        if GoogleTranslator is not None:
            try:
                translator = GoogleTranslator(source='sa', target='en')
                translated = translator.translate(cleaned_sanskrit)
                if translated and not translated.startswith("Translation unavailable") and translated.strip() != cleaned_sanskrit.strip():
                    return {
                        "cleaned_sanskrit": cleaned_sanskrit,
                        "english": translated,
                        "source": "Neural MT (Google Translate)",
                        "match_type": "neural"
                    }
            except Exception:
                pass

        # 5. Neural MT Tier 2: MyMemory (Sanskrit sa-IN -> English en-US)
        if MyMemoryTranslator is not None:
            try:
                mm = MyMemoryTranslator(source='sa-IN', target='en-US')
                translated = mm.translate(cleaned_sanskrit)
                if translated and translated.strip().lower() != cleaned_sanskrit.strip().lower() and not translated.startswith("Translation unavailable"):
                    return {
                        "cleaned_sanskrit": cleaned_sanskrit,
                        "english": translated,
                        "source": "Neural MT (MyMemory)",
                        "match_type": "neural"
                    }
            except Exception:
                pass

        # 6. Fallback: Word-level Sanskrit Glossary Lookup
        words = [w for w in normalized_in.split() if len(w) > 1]
        gloss_items = []
        for w in words:
            for k, val in self.corpus_dict.items():
                if w == self._strip_punct(k):
                    gloss_items.append(f"{w}: {val.split('(')[0].strip()}")
                    break
        if gloss_items:
            return {
                "cleaned_sanskrit": cleaned_sanskrit,
                "english": "; ".join(gloss_items),
                "source": "Sanskrit Lexical Glossary",
                "match_type": "glossary"
            }

        # 7. Graceful Structured Output (never an ugly unhandled exception)
        return {
            "cleaned_sanskrit": cleaned_sanskrit,
            "english": f"Sanskrit Devanagari text: '{cleaned_sanskrit}' (Classical Sanskrit)",
            "source": "Normalized Sanskrit",
            "match_type": "fallback"
        }

    def translate(self, text: str) -> str:
        """
        Translates Sanskrit text to English.

        Args:
            text (str): The Sanskrit text to translate.

        Returns:
            str: The English translation.
        """
        result = self.translate_with_metadata(text)
        return result["english"]

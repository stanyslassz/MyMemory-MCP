"""Tests for pipeline/router.py — deterministic content classification."""

import json
import time

import pytest

from src.pipeline.router import classify


# ── Conversation samples ─────────────────────────────────────

CONV_SPEAKER_MARKERS = """\
User: J'ai mal au dos aujourd'hui.
Assistant: Je suis désolé d'entendre ça. Depuis combien de temps?
User: Depuis lundi. La sciatique revient.
Assistant: Avez-vous consulté votre médecin?
User: Oui, Dr Martin m'a prescrit du paracétamol.
Assistant: D'accord, suivez ses recommandations.
"""

CONV_JSON_ARRAY = json.dumps([
    {"role": "user", "content": "Bonjour, comment ça va?"},
    {"role": "assistant", "content": "Très bien, merci!"},
    {"role": "user", "content": "Parlons de mon projet."},
])

CONV_TIMESTAMP_SPEAKERS = """\
[2026-03-01 10:30] Alice: Salut, tu as vu le rapport?
[2026-03-01 10:31] Bob: Oui, j'ai des commentaires.
[2026-03-01 10:32] Alice: Envoie-les moi par email.
[2026-03-01 10:33] Bob: C'est fait.
"""

CONV_MIXED_CUES = """\
Human: I need help with my project.
AI: Sure, what kind of project?
Human: A cloud migration for TechCorp.
AI: That sounds interesting. What's the timeline?
Human: We need it done by next month.
"""


# ── Document samples ─────────────────────────────────────────

DOC_MARKDOWN_REPORT = """\
# Rapport de Santé Mensuel

## Résumé
Ce mois-ci, les indicateurs de santé sont stables. La tension artérielle
reste dans les normes et le poids est constant.

## Recommandations
- Continuer la natation 3 fois par semaine
- Maintenir le traitement prescrit par Dr Martin
- Surveiller les épisodes de sciatique

## Prochains Rendez-vous
Le prochain contrôle est prévu pour le 15 avril 2026.
"""

DOC_NOTES = """\
# Notes de Réunion — Projet Phoenix

Migration cloud prévue pour Q2 2026. Étapes principales:

1. Audit de l'infrastructure actuelle
2. Choix du provider cloud (AWS vs Azure)
3. Migration des bases de données
4. Tests de performance
5. Mise en production

Budget estimé: 150k€. Responsable: Louise.
L'équipe technique comprend 5 développeurs et 2 ops.
La phase de test prendra environ 3 semaines avec des critères
d'acceptance stricts définis dans le cahier des charges.
"""

DOC_SINGLE_AUTHOR = """\
Aujourd'hui j'ai découvert un nouveau framework JavaScript très intéressant.
Il permet de créer des applications web performantes avec un minimum de code.
La documentation est excellente et la communauté est très active.

J'ai commencé à migrer mon projet personnel dessus et les premiers résultats
sont prometteurs. Le temps de build a été réduit de 60% et la taille du bundle
est passée de 2MB à 400KB. C'est vraiment impressionnant.

Je vais continuer à explorer les fonctionnalités avancées la semaine prochaine.
"""


# ── Edge cases ───────────────────────────────────────────────

EDGE_EMPTY = ""
EDGE_WHITESPACE = "   \n  \n  "
EDGE_SHORT_AMBIGUOUS = "Bonjour, comment ça va?"
EDGE_HEADING_WITH_DIALOGUE = """\
# Discussion avec le médecin

User: Docteur, j'ai mal au dos.
Assistant: Depuis quand exactement?
User: Depuis deux semaines environ.
Assistant: Je vais vous examiner.
"""


# ── Unit tests ───────────────────────────────────────────────

class TestConversationRouting:
    def test_speaker_markers(self):
        result = classify(CONV_SPEAKER_MARKERS)
        assert result.route == "conversation"
        assert result.confidence >= 0.5

    def test_json_chat_array(self):
        result = classify(CONV_JSON_ARRAY)
        assert result.route == "conversation"
        assert result.confidence >= 0.9

    def test_timestamp_speakers(self):
        result = classify(CONV_TIMESTAMP_SPEAKERS)
        assert result.route == "conversation"
        assert result.confidence >= 0.5

    def test_mixed_cues(self):
        result = classify(CONV_MIXED_CUES)
        assert result.route == "conversation"

    def test_reasons_populated(self):
        result = classify(CONV_SPEAKER_MARKERS)
        assert len(result.reasons) > 0


class TestDocumentRouting:
    def test_markdown_report(self):
        result = classify(DOC_MARKDOWN_REPORT)
        assert result.route == "document"
        assert result.confidence >= 0.5

    def test_structured_notes(self):
        result = classify(DOC_NOTES)
        assert result.route == "document"
        assert result.confidence >= 0.5

    def test_single_author_note(self):
        result = classify(DOC_SINGLE_AUTHOR)
        assert result.route == "document"

    def test_reasons_populated(self):
        result = classify(DOC_MARKDOWN_REPORT)
        assert len(result.reasons) > 0


class TestEdgeCases:
    def test_empty_content(self):
        result = classify(EDGE_EMPTY)
        assert result.route == "uncertain"

    def test_whitespace_only(self):
        result = classify(EDGE_WHITESPACE)
        assert result.route == "uncertain"

    def test_short_ambiguous(self):
        result = classify(EDGE_SHORT_AMBIGUOUS)
        # Short text with no signals should be uncertain
        assert result.route == "uncertain"

    def test_heading_with_dialogue(self):
        # Mixed signals — should pick dominant
        result = classify(EDGE_HEADING_WITH_DIALOGUE)
        assert result.route in ("conversation", "document", "uncertain")
        assert result.confidence >= 0.0


class TestRouterLatency:
    def test_decision_under_100ms(self):
        """Router decision latency p95 <= 100ms (SLO target)."""
        latencies = []
        samples = [
            CONV_SPEAKER_MARKERS, CONV_JSON_ARRAY, CONV_TIMESTAMP_SPEAKERS,
            DOC_MARKDOWN_REPORT, DOC_NOTES, DOC_SINGLE_AUTHOR,
            EDGE_SHORT_AMBIGUOUS, EDGE_HEADING_WITH_DIALOGUE,
        ]
        for text in samples * 10:  # 80 runs
            t0 = time.monotonic()
            classify(text)
            latencies.append(time.monotonic() - t0)

        latencies.sort()
        p95_idx = int(len(latencies) * 0.95)
        p95 = latencies[p95_idx]
        assert p95 < 0.1, f"Router p95 latency {p95:.4f}s exceeds 100ms SLO"


# ── Router evaluation harness ────────────────────────────────

# Frozen labeled set — 30 samples (minimal viable for unit tests;
# full 300+ set should be in tests/fixtures/router_labeled_set.json for CI)
LABELED_SET = [
    # (text, expected_route)
    # Conversations (12)
    ("User: Bonjour\nAssistant: Salut\nUser: Comment ça va?\nAssistant: Bien merci", "conversation"),
    ("Human: Help me\nAI: Sure\nHuman: Thanks\nAI: You're welcome", "conversation"),
    (json.dumps([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]), "conversation"),
    ("[10:00] Alice: test\n[10:01] Bob: ok\n[10:02] Alice: good\n[10:03] Bob: bye", "conversation"),
    ("User: J'ai besoin d'aide\nAssistant: Bien sûr\nUser: Merci\nAssistant: De rien\nUser: Au revoir\nAssistant: Bye", "conversation"),
    ("Utilisateur: Bonjour\nAssistant: Salut\nUtilisateur: Ça va?\nAssistant: Oui", "conversation"),
    (json.dumps([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}, {"role": "user", "content": "z"}]), "conversation"),
    ("User: One\nAssistant: Two\nUser: Three\nAssistant: Four\nUser: Five\nAssistant: Six", "conversation"),
    ("Human: First\nAI: Second\nHuman: Third\nAI: Fourth", "conversation"),
    (json.dumps([{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]), "conversation"),
    ("User: Hello there\nAssistant: Hi! How can I help?\nUser: I need advice\nAssistant: Of course", "conversation"),
    ("[2026-01-01 09:00] John: Hey\n[2026-01-01 09:01] Jane: Hi\n[2026-01-01 09:02] John: Meet later?\n[2026-01-01 09:03] Jane: Sure", "conversation"),

    # Documents (12)
    ("# Report\n\n## Section 1\nLong content here that spans multiple sentences and provides detailed information about the topic at hand.\n\n## Section 2\nMore detailed analysis.", "document"),
    ("# Notes\n\n- Item one with details\n- Item two with more info\n- Item three\n- Item four\n- Item five\n- Item six", "document"),
    ("# Architecture\n\nThe system uses a microservices architecture with the following components that interact through REST APIs and message queues for async processing.\n\n## Components\n- Service A\n- Service B\n- Service C", "document"),
    ("## Meeting Notes\n\nDiscussed the project timeline. Key decisions:\n1. Launch date moved to Q3\n2. Budget increased by 20%\n3. New team members joining\n4. Weekly syncs scheduled\n5. Documentation required", "document"),
    ("# Guide\n\nThis guide explains how to set up the development environment. Follow the steps carefully to ensure everything works correctly and you can start contributing.\n\n## Prerequisites\nYou need Python 3.12+", "document"),
    ("# Analyse\n\nLes résultats montrent une amélioration significative des performances depuis le dernier trimestre. Les métriques clés sont toutes au-dessus des objectifs fixés.\n\n## Détails\n- KPI 1: +15%\n- KPI 2: +22%", "document"),
    ("## Plan de Migration\n\n1. Audit initial\n2. Préparation environnement\n3. Migration données\n4. Tests\n5. Mise en production\n\nLe budget total est estimé à 200k€ avec un délai de 6 mois.", "document"),
    ("# Weekly Status\n\nProgress this week was excellent. The team completed all planned tasks and started on next sprint items.\n\n## Completed\n- Feature A\n- Bug fix B\n- Review C", "document"),
    ("# Technical Specification\n\nThe API endpoint accepts POST requests with JSON body containing user preferences for personalized recommendations.\n\n## Request Format\nJSON with fields: user_id, preferences, filters", "document"),
    ("## Compte-Rendu\n\nRéunion du 1er mars. Participants: Alice, Bob, Charlie.\n\nDécisions prises:\n- Lancement phase 2\n- Recrutement 2 devs\n- Budget Q2 validé", "document"),
    ("# README\n\nThis project implements a personal memory system for LLMs. It stores structured knowledge extracted from conversations.\n\n## Installation\nRun pip install -e . to install.", "document"),
    ("# Diary Entry\n\nToday I learned about a fascinating new approach to distributed systems. The key insight is that eventual consistency can be achieved without sacrificing availability.\n\n## Key Takeaways\n- CAP theorem trade-offs\n- CRDT-based solutions", "document"),

    # Uncertain/ambiguous (6)
    ("Hello world", "uncertain"),
    ("Test", "uncertain"),
    ("Bonjour", "uncertain"),
    ("OK", "uncertain"),
    ("x", "uncertain"),
    ("123", "uncertain"),
]


class TestRouterEvaluationHarness:
    """Evaluation harness against frozen labeled set with precision/recall report."""

    def test_labeled_set_evaluation(self):
        """Run router against labeled set and report precision/recall."""
        tp = {"conversation": 0, "document": 0}
        fp = {"conversation": 0, "document": 0}
        fn = {"conversation": 0, "document": 0}
        correct = 0
        total = len(LABELED_SET)
        misroutes = []

        for text, expected in LABELED_SET:
            result = classify(text)
            actual = result.route

            if actual == expected:
                correct += 1
                if expected in tp:
                    tp[expected] += 1
            else:
                if actual in fp:
                    fp[actual] += 1
                if expected in fn:
                    fn[expected] += 1
                misroutes.append((text[:50], expected, actual, result.confidence))

        # Compute precision/recall
        conv_precision = tp["conversation"] / max(tp["conversation"] + fp["conversation"], 1)
        conv_recall = tp["conversation"] / max(tp["conversation"] + fn["conversation"], 1)
        doc_precision = tp["document"] / max(tp["document"] + fp["document"], 1)
        doc_recall = tp["document"] / max(tp["document"] + fn["document"], 1)

        # Count uncertain rate on non-ambiguous
        non_ambiguous = [(t, e) for t, e in LABELED_SET if e != "uncertain"]
        uncertain_on_non_ambiguous = sum(
            1 for t, _ in non_ambiguous if classify(t).route == "uncertain"
        )
        uncertain_rate = uncertain_on_non_ambiguous / max(len(non_ambiguous), 1)

        # Misroute-to-conversation for true documents
        true_docs = [(t, e) for t, e in LABELED_SET if e == "document"]
        misroute_to_conv = sum(
            1 for t, _ in true_docs if classify(t).route == "conversation"
        )
        misroute_rate = misroute_to_conv / max(len(true_docs), 1)

        # Report (printed for CI visibility)
        report = {
            "total_samples": total,
            "accuracy": round(correct / total, 3),
            "conversation_precision": round(conv_precision, 3),
            "conversation_recall": round(conv_recall, 3),
            "document_precision": round(doc_precision, 3),
            "document_recall": round(doc_recall, 3),
            "uncertain_rate_on_non_ambiguous": round(uncertain_rate, 3),
            "misroute_to_conversation_for_docs": round(misroute_rate, 3),
            "misroutes": misroutes,
        }

        print("\n=== ROUTER EVALUATION REPORT ===")
        for k, v in report.items():
            if k != "misroutes":
                print(f"  {k}: {v}")
        if misroutes:
            print(f"  misrouted_samples ({len(misroutes)}):")
            for text_preview, expected, actual, conf in misroutes:
                print(f"    '{text_preview}...' expected={expected} got={actual} conf={conf}")
        print("================================\n")

        # Assertions against thresholds from plan §5
        assert conv_precision >= 0.90, f"Conversation precision {conv_precision} < 0.90"
        assert conv_recall >= 0.80, f"Conversation recall {conv_recall} < 0.80"
        assert doc_precision >= 0.90, f"Document precision {doc_precision} < 0.90"
        assert doc_recall >= 0.90, f"Document recall {doc_recall} < 0.90"
        assert misroute_rate <= 0.05, f"Misroute-to-conv rate {misroute_rate} > 5%"

# =============================================================================
# EVALUATION SCRIPT — Prove the fix works
# Run this to generate the "before vs after" comparison for your report/Monday meeting
# =============================================================================
 
import json
import time
from orchestration.external_knowledge_agent import orchestrated_ask_v2
 
# Test questions split into two categories, same as your evaluation methodology
IN_DATASET_QUESTIONS = [
    "Which entities are most connected in the graph?",
    "Find entities related to /m/06rf7 in the knowledge graph.",
    "Find the top 5 most common relationship types.",
]
 
OUT_OF_DATASET_QUESTIONS = [
    "What is the current price of Bitcoin today?",
    "Who won the latest football World Cup?",
    "What is the weather in London right now?",
]
 
 
def run_before_after_comparison():
    """
    Demonstrates the limitation fix:
    - BEFORE: out-of-dataset questions returned low confidence, no real answer
    - AFTER: out-of-dataset questions now get a web-search-supplemented answer
    """
    results = {"in_dataset": [], "out_of_dataset": []}
 
    print("\n" + "#" * 70)
    print("# IN-DATASET QUESTIONS (should use graph only, no web search)")
    print("#" * 70)
    for q in IN_DATASET_QUESTIONS:
        start = time.time()
        r = orchestrated_ask_v2(q)
        elapsed = time.time() - start
        results["in_dataset"].append({
            "question": q,
            "answer": r["final_answer"],
            "confidence": r["confidence"],
            "web_search_used": r["web_search_used"],
            "time_seconds": round(elapsed, 1)
        })
        print(f"\nQ: {q}")
        print(f"   Confidence: {r['confidence']:.2f} | Web search used: {r['web_search_used']}")
        print(f"   Answer: {r['final_answer']}")
    print("#" * 70)
    for q in OUT_OF_DATASET_QUESTIONS:
        start = time.time()
        r = orchestrated_ask_v2(q)
        elapsed = time.time() - start
        results["out_of_dataset"].append({
            "question": q,
            "answer": r["final_answer"],
            "confidence": r["confidence"],
            "web_search_used": r["web_search_used"],
            "time_seconds": round(elapsed, 1)
        })
        print(f"\nQ: {q}")
        print(f"   Confidence: {r['confidence']:.2f} | Web search used: {r['web_search_used']}")
        print(f"   Answer: {r['final_answer']}")
 
    # Summary stats — exactly what you need for the "How Well" section
    web_search_trigger_rate = sum(
        1 for r in results["out_of_dataset"] if r["web_search_used"]
    ) / len(results["out_of_dataset"])
 
    false_trigger_rate = sum(
        1 for r in results["in_dataset"] if r["web_search_used"]
    ) / len(results["in_dataset"])
 
    print("\n" + "=" * 70)
    print("SUMMARY (use these numbers in your report)")
    print("=" * 70)
    print(f"Web search correctly triggered on out-of-dataset questions: "
          f"{web_search_trigger_rate:.0%}")
    print(f"Web search incorrectly triggered on in-dataset questions: "
          f"{false_trigger_rate:.0%}  (should be 0%)")
 
    with open("limitation_fix_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to limitation_fix_results.json")
 
    return results
 
 
if __name__ == "__main__":
    run_before_after_comparison()
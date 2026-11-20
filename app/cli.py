
"""Command-line interface for Legal Sentence Game."""

import argparse
from app.core import (
    milestone_1,
    milestone_2,
    milestone_4,
    milestone_5,
    predict_next
)


def play_game():
    print("\n===================================")
    print("       LEGAL SENTENCE GAME")
    print("===================================")

    corpus = milestone_1()
    model = milestone_2(corpus)
    sentences = milestone_4()

    score = 0
    attempted = 0

    for number, partial in enumerate(sentences, start=1):
        print(f"\nQuestion {number}/{len(sentences)}")
        print("Complete this legal sentence:")
        print(partial)

        suggestions = predict_next(model, partial)

        if suggestions:
            print("\nSuggested next words:")
            print(", ".join(suggestions))
        else:
            print("\nNo next-word suggestions available.")

        answer = input("\nComplete the sentence: ").strip()

        if not answer:
            print("You skipped this question.")
            continue

        # Combine the incomplete sentence and user's answer.
        if answer.lower().startswith(partial.lower()):
            full_sentence = answer
        else:
            full_sentence = partial + " " + answer

        print("\nYour complete sentence:")
        print(full_sentence)

        attempted += 1
        result = milestone_5(full_sentence)

        print("\n--- Feedback ---")
        print(result["feedback"])

        if result["category"] != "Unknown":
            score += 1

    print("\n===================================")
    print("GAME SUMMARY")
    print(f"Questions: {len(sentences)}")
    print(f"Attempted: {attempted}")
    print(f"Skipped: {len(sentences) - attempted}")
    print(f"Successfully classified: {score}")
    print("===================================")
    print("Thank you for playing!")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="legal-sentence-game",
        description="Practice legal sentence drafting."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hello", help="Check project setup")
    sub.add_parser("play", help="Start the legal sentence game")

    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Legal Sentence Game is ready!")
    elif args.command == "play":
        play_game()


if __name__ == "__main__":
    main()

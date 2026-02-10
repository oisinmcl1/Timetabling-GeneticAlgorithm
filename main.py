import sys
import argparse
import random
import copy
from typing import List, Tuple, Sequence


def read_instance(filename: str) -> Tuple[int, int, int, List[List[int]]]:
    """Read an instance file and return (num_exams, num_timeslots, num_students, student_exams).
    student_exams is a list of lists where each inner list contains exam indices (0-based)
    that the student is registered for.
    """
    student_exams: List[List[int]] = []

    with open(filename, "r", encoding="utf-8") as f:
        header_line = None
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            header_line = line
            break
        if header_line is None:
            raise ValueError("File is empty or contains only comments/blank lines")

        parts = header_line.split()
        if len(parts) < 3:
            raise ValueError("Header must contain at least three integers: n k m")

        try:
            num_exams, num_timeslots, num_students = map(int, parts[:3])
        except ValueError as e:
            raise ValueError("Header values must be integers: n k m") from e

        # read the student rows
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            try:
                ints = [int(v) for v in vals]
            except ValueError:
                raise ValueError(f"Student row contains non-integer values: {line}")

            exams = [idx for idx, val in enumerate(ints[:num_exams]) if val == 1]
            student_exams.append(exams)

    return num_exams, num_timeslots, num_students, student_exams



def parse_args():
    p = argparse.ArgumentParser(description="Exam Timetabling GA")
    p.add_argument("instance", nargs="?", default="/Users/ciarangray/Desktop/4th year/AI/A1/test_case1.txt", help="instance file (default: /Users/ciarangray/Desktop/4th year/AI/A1/test_case1.txt)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        num_exams, num_timeslots, num_students, student_exams = read_instance(args.instance)
    except Exception as e:
        print(f"Error reading instance '{args.instance}': {e}")
        sys.exit(1)

    print(f"Instance: exams={num_exams}, timeslots={num_timeslots}, students={num_students}")
    print(f"Parsed {len(student_exams)} student rows")


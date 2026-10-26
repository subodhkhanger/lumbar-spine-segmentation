import argparse
import json
from pathlib import Path
import sys

from nibabel.filebasedimages import ImageFileError

from .artifacts import file_hash, reject_path_overlap, write_json
from .geometry import load_volume
from .prepare import prepare_source, prepare_verse, source_metadata, write_prepared


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Lumbar CT CPU audit and case preparation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser(
        "audit", help="audit a CT header and scaled intensities"
    )
    audit.add_argument("input", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    source = commands.add_parser(
        "prepare-source", help="prepare a source case using official metadata"
    )
    source.add_argument("case_dir", type=Path)

    source.add_argument("--case-id", required=True)
    source.add_argument("--metadata", type=Path, required=True)

    source.add_argument("--output", type=Path, required=True)
    verse = commands.add_parser(
        "prepare-verse", help="format check of external CT/reference; no model scoring"
    )
    verse.add_argument("ct", type=Path)
    verse.add_argument("reference", type=Path)

    verse.add_argument("--case-id", required=True)
    verse.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:

        if args.command == "audit":
            reject_path_overlap(args.input, args.output)
            volume = load_volume(args.input)
            record = {
                "status": "passed",
                "audit": volume.audit,
                "ct_sha256": file_hash(args.input),
            }
            write_json(args.output, record)

        elif args.command == "prepare-source":
            metadata = source_metadata(args.metadata)
            if args.case_id not in metadata:
                raise ValueError("case ID not in official metadata")
            prepared = prepare_source(
                args.case_dir,
                case_id=args.case_id,
                split=metadata[args.case_id]["split"],
            )
            prepared.record["metadata_sha256"] = file_hash(args.metadata)
            write_prepared(prepared, args.output)

        else:
            prepared = prepare_verse(args.ct, args.reference, case_id=args.case_id)
            write_prepared(prepared, args.output)

        print(json.dumps({"status": "completed", "output": str(args.output.resolve())}))
        return 0


    except (ValueError, OSError, KeyError, ImageFileError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import yaml
from docling_core.types.doc import ImageRefMode

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.document import ConversionResult
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

_log = logging.getLogger(__name__)
USE_V2 = True # use modern Docling export formats
USE_LEGACY = False

""" 
EXAMPLE WITH SINGLE DOCUMENT ----------------------------------------------------------------------------------------------
source = "clinical_pathways_pdfs/agitation_education_module_-_9.6.23.pdf"  # file path or URL
converter = DocumentConverter()
doc = converter.convert(source).document

print(doc.export_to_markdown())  # output: "### Docling Technical Report[...]"
END ----------------------------------------------------------------------------------------------------------------------
"""

# Input documents and export in various formats
def export_documents( 
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = conv_res.input.file.stem

            if USE_V2:
                # Recommended modern Docling exports. These helpers mirror the
                # lower-level "export_to_*" methods used below, but handle
                # common details like image handling.
                conv_res.document.save_as_json(
                    output_dir / f"{doc_filename}.json",
                    image_mode=ImageRefMode.PLACEHOLDER,
                )
                # conv_res.document.save_as_html(
                #     output_dir / f"{doc_filename}.html",
                #     image_mode=ImageRefMode.EMBEDDED,
                # )
                # conv_res.document.save_as_doctags(
                #     output_dir / f"{doc_filename}.doctags.txt"
                # )
                conv_res.document.save_as_markdown(
                    output_dir / f"{doc_filename}.md",
                    image_mode=ImageRefMode.PLACEHOLDER,
                )
                conv_res.document.save_as_markdown(
                    output_dir / f"{doc_filename}.txt",
                    image_mode=ImageRefMode.PLACEHOLDER,
                    strict_text=True,
                )

                # Export Docling document format to YAML:
                with (output_dir / f"{doc_filename}.yaml").open("w") as fp:
                    fp.write(yaml.safe_dump(conv_res.document.export_to_dict()))

                # # Export Docling document format to doctags:
                # with (output_dir / f"{doc_filename}.doctags.txt").open("w") as fp:
                #     fp.write(conv_res.document.export_to_doctags())

                # Export Docling document format to markdown:
                with (output_dir / f"{doc_filename}.md").open("w") as fp:
                    fp.write(conv_res.document.export_to_markdown())

                # Export Docling document format to text:
                with (output_dir / f"{doc_filename}.txt").open("w") as fp:
                    fp.write(conv_res.document.export_to_markdown(strict_text=True))

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            _log.info(
                f"Document {conv_res.input.file} was partially converted with the following errors:"
            )
            for item in conv_res.errors:
                _log.info(f"\t{item.error_message}")
            partial_success_count += 1
        else:
            _log.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    _log.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    return success_count, partial_success_count, failure_count

def main():
    logging.basicConfig(level=logging.INFO)

    # Location of sample PDFs used by this example. If your checkout does not
    # include test data, change `data_folder` or point `input_doc_paths` to
    # your own files.
    data_folder = Path(__file__).parent / "clinical_pathways_pdfs/"
    input_doc_paths = [
        data_folder / "anaphylaxis_-_1.16.25.pdf",
        data_folder / "anaphylaxis_educational_module_1.16.25.pdf",
    ]

    # Configure the PDF pipeline. Enabling page image generation improves HTML
    # previews (embedded images) but adds processing time.
    pipeline_options = PdfPipelineOptions()

    """ ENABLE ON BETTER MACHINE """
    # pipeline_options.generate_page_images = True 

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=DoclingParseV4DocumentBackend
            )
        }
    )

    start_time = time.time()

    # Convert all inputs. Set `raises_on_error=False` to keep processing other
    # files even if one fails; errors are summarized after the run.
    conv_results = doc_converter.convert_all(
        input_doc_paths,
        raises_on_error=False,  # to let conversion run through all and examine results at the end
    )
    # Write outputs to ./scratch and log a summary.
    _success_count, _partial_success_count, failure_count = export_documents(
        conv_results, output_dir=Path("scratch")
    )

    end_time = time.time() - start_time

    _log.info(f"Document conversion complete in {end_time:.2f} seconds.")

    if failure_count > 0:
        raise RuntimeError(
            f"The example failed converting {failure_count} on {len(input_doc_paths)}."
        )


if __name__ == "__main__":
    main()
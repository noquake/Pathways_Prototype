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
source = "/data/agitation_education_module_-_9.6.23.pdf"  # file path or URL
converter = DocumentConverter()
doc = converter.convert(source).document

print(doc.export_to_markdown())  # output: "### Docling Technical Report[...]"
END ----------------------------------------------------------------------------------------------------------------------
"""

def save_document_json(document, output_path: Path, image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER):
    document.save_as_json(output_path, image_mode=image_mode)

def save_document_markdown(document, output_path: Path, image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER):
    document.save_as_markdown(output_path, image_mode=image_mode)

def save_document_text(document, output_path: Path, image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER):
    document.save_as_markdown(output_path, image_mode=image_mode, strict_text=True)

def save_document_yaml(document, output_path: Path):
    with output_path.open("w") as fp:
        fp.write(yaml.safe_dump(document.export_to_dict()))

# save each conversion result into a directory of all documents formats
def save_all_document_formats(conv_res: ConversionResult, output_dir: Path):
    doc_filename = conv_res.input.file.stem
    
    save_document_json(
        conv_res.document,
        output_dir / f"{doc_filename}.json",
        image_mode=ImageRefMode.PLACEHOLDER
    )
    
    save_document_markdown(
        conv_res.document,
        output_dir / f"{doc_filename}.md",
        image_mode=ImageRefMode.PLACEHOLDER
    )
    
    save_document_text(
        conv_res.document,
        output_dir / f"{doc_filename}.txt",
        image_mode=ImageRefMode.PLACEHOLDER
    )
    
    save_document_yaml(
        conv_res.document,
        output_dir / f"{doc_filename}.yaml"
    )
    
    # save_document_html(
    #     conv_res.document,
    #     output_dir / f"{doc_filename}.html",
    #     image_mode=ImageRefMode.EMBEDDED
    # )

    # save_document_doctags(
    #     conv_res.document,
    #     output_dir / f"{doc_filename}.doctags.txt"
    # )

# True if successful, False otherwise
def handle_successful_conversion(conv_res: ConversionResult, output_dir: Path) -> bool:
    try:
        save_all_document_formats(conv_res, output_dir)
        return True
    except Exception as e:
        _log.error(f"Error saving document {conv_res.input.file}: {e}")
        return False

# check if partial success has errors and log them
def handle_partial_success(conv_res: ConversionResult):
    _log.info(
        f"Document {conv_res.input.file} was partially converted with the following errors:"
    )
    for item in conv_res.errors:
        _log.info(f"\t{item.error_message}")

# Gather info on failed conversions
def handle_conversion_failure(conv_res: ConversionResult):
    _log.info(f"Document {conv_res.input.file} failed to convert.")

# Get a summary of conversion results
def conversion_summary(success_count: int, partial_success_count: int, failure_count: int):
    total = success_count + partial_success_count + failure_count
    _log.info(
        f"Processed {total} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )

def export_conversion_summary( 
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
):
    """
    Export converted documents to various formats.
    
    Args:
        conv_results: Iterable of conversion results
        output_dir: Directory where output files should be saved
    
    Returns:
        Tuple of (success_count, partial_success_count, failure_count)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            if handle_successful_conversion(conv_res, output_dir):
                success_count += 1
        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            handle_partial_success(conv_res)
            partial_success_count += 1
        else:
            handle_conversion_failure(conv_res)
            failure_count += 1

    conversion_summary(success_count, partial_success_count, failure_count)
    return success_count, partial_success_count, failure_count

def main():
    logging.basicConfig(level=logging.INFO)

    # Location of sample PDFs used by this example. If your checkout does not
    # include test data, change `data_folder` or point `input_doc_paths` to
    # your own files.
    
    data_folder = Path(__file__).parent / "data/"

    input_doc_paths = list(data_folder.glob("*")) 

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
    _success_count, _partial_success_count, failure_count = export_conversion_summary(
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
# import os
# import json
# import base64
# from io import BytesIO
# from PyPDF2 import PdfReader, PdfWriter
# import google.auth
# from google.auth.transport.requests import AuthorizedSession
# import concurrent.futures

# def pdf_page_to_bytes(pdf_reader, page_number):
#     writer = PdfWriter()
#     writer.add_page(pdf_reader.pages[page_number])
#     output_stream = BytesIO()
#     writer.write(output_stream)
#     return output_stream.getvalue()

# def get_text_from_text_anchor(full_text, text_anchor):
#     if "textSegments" not in text_anchor:
#         return ""
#     segments = text_anchor["textSegments"]
#     extracted_text = ""
#     for segment in segments:
#         start_index = int(segment.get("startIndex", 0))
#         end_index = int(segment.get("endIndex", 0))
#         extracted_text += full_text[start_index:end_index]
#     return extracted_text.strip()

# def process_document_rest(project_id, location, processor_id, pdf_bytes):
#     credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
#     authed_session = AuthorizedSession(credentials)

#     url = f"https://{location}-documentai.googleapis.com/v1/projects/{project_id}/locations/{location}/processors/{processor_id}:process"

#     encoded_content = base64.b64encode(pdf_bytes).decode("utf-8")

#     body = {
#         "rawDocument": {
#             "content": encoded_content,
#             "mimeType": "application/pdf"
#         }
#     }

#     response = authed_session.post(url, json=body)

#     if response.status_code == 200:
#         response_json = response.json()
#         document = response_json.get("document", {})
#         full_text = document.get("text", "")
#         pages = document.get("pages", [])

#         for page in pages:
#             text_anchor = page.get("layout", {}).get("textAnchor", {})
#             page_text = get_text_from_text_anchor(full_text, text_anchor)
#             page["text"] = page_text

#         return pages
#     else:
#         print(f" Error in process_document_rest: {response.status_code} - {response.text}")
#         response.raise_for_status()

# def process_pdf_and_return_result(project_id, location, processor_id, input_pdf):
#     pdf_reader = PdfReader(input_pdf)
#     print("pdf_reader in process_pdf_and_return_result" , pdf_reader)
#     total_pages = len(pdf_reader.pages)
#     all_pages = []

#     def process_single_page(page_idx):
#         print(f"Processing page {page_idx + 1} of {os.path.basename(input_pdf)} in process_single_page")
#         pdf_bytes = pdf_page_to_bytes(pdf_reader, page_idx)
#         pages = process_document_rest(project_id, location, processor_id, pdf_bytes)
#         for page in pages:
#             page["pageNumber"] = page_idx + 1
#         return pages

#     with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
#         results = executor.map(process_single_page, range(total_pages))

#     for pages in results:
#         all_pages.extend(pages)

#     return {
#         "document": {
#             "pages": all_pages
#         }
#     }

# def process_folder_of_pdfs(project_id, location, processor_id, folder_path):
#     for filename in os.listdir(folder_path):
#         if filename.lower().endswith(".pdf"):
#             input_pdf = os.path.join(folder_path, filename)
#             try:
#                 print(f"\n📄 Processing file: {filename} in process_folder_of_pdfs")
#                 result = process_pdf_and_return_result(project_id, location, processor_id, input_pdf)

#                 # Save result to a JSON file (same name as PDF)
#                 base_name = os.path.splitext(filename)[0]
#                 output_json = os.path.join(folder_path, base_name + ".json")
#                 with open(output_json, "w", encoding="utf-8") as f:
#                     json.dump(result, f, indent=2, ensure_ascii=False)

#                 print(f"Saved to : {output_json} using process_folder_of_pdfs")
#             except Exception as e:
#                 print(f" Error processing in process_folder_of_pdfs {filename}: {e} ")



# # Example usage
# if __name__ == "__main__":
#     PROJECT_ID = "935917861333"
#     LOCATION = "us"
#     PROCESSOR_ID = "6078b90020675fba"
#     FOLDER_PATH = r"D:\pdf_chtbot\ai-assistant\women_data_pdf"  # Folder containing PDFs

#     process_folder_of_pdfs(PROJECT_ID, LOCATION, PROCESSOR_ID, FOLDER_PATH)



import os
import json
import base64
import time
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter
import google.auth
from google.auth.transport.requests import AuthorizedSession
import concurrent.futures
from requests.exceptions import RequestException, SSLError, ConnectionError, HTTPError

MAX_RETRIES = 6
TIMEOUT_SECONDS = 300


def pdf_page_to_bytes(pdf_reader, page_number):
    writer = PdfWriter()
    writer.add_page(pdf_reader.pages[page_number])
    output_stream = BytesIO()
    writer.write(output_stream)
    return output_stream.getvalue()


def get_text_from_text_anchor(full_text, text_anchor):
    if "textSegments" not in text_anchor:
        return ""
    segments = text_anchor["textSegments"]
    extracted_text = ""
    for segment in segments:
        start_index = int(segment.get("startIndex", 0))
        end_index = int(segment.get("endIndex", 0))
        extracted_text += full_text[start_index:end_index]
    return extracted_text.strip()


def process_document_rest(project_id, location, processor_id, pdf_bytes):
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    authed_session = AuthorizedSession(credentials)

    url = f"https://{location}-documentai.googleapis.com/v1/projects/{project_id}/locations/{location}/processors/{processor_id}:process"
    encoded_content = base64.b64encode(pdf_bytes).decode("utf-8")
    body = {
        "rawDocument": {
            "content": encoded_content,
            "mimeType": "application/pdf"
        }
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = authed_session.post(url, json=body, timeout=TIMEOUT_SECONDS)
            if response.status_code == 200:
                response_json = response.json()
                document = response_json.get("document", {})
                full_text = document.get("text", "")
                pages = document.get("pages", [])

                for page in pages:
                    text_anchor = page.get("layout", {}).get("textAnchor", {})
                    page_text = get_text_from_text_anchor(full_text, text_anchor)
                    page["text"] = page_text

                return pages
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                response.raise_for_status()

        except (SSLError, ConnectionError, HTTPError, RequestException) as e:
            print(f"⚠️ Attempt {attempt} failed due to: {type(e).__name__} - {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                raise


def process_pdf_and_return_result(project_id, location, processor_id, input_pdf):
    try:
        pdf_reader = PdfReader(input_pdf)
    except Exception as e:
        print(f"❌ Failed to read PDF {input_pdf}: {e}")
        return None

    total_pages = len(pdf_reader.pages)
    all_pages = []

    def process_single_page(page_idx):
        print(f"🔄 Processing page {page_idx + 1} of {os.path.basename(input_pdf)}")
        try:
            pdf_bytes = pdf_page_to_bytes(pdf_reader, page_idx)
            pages = process_document_rest(project_id, location, processor_id, pdf_bytes)
            for page in pages:
                page["pageNumber"] = page_idx + 1
            return pages
        except Exception as e:
            print(f"❌ Failed page {page_idx + 1}: {e}")
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(process_single_page, range(total_pages))

    for pages in results:
        all_pages.extend(pages)

    return {"document": {"pages": all_pages}}


def process_folder_of_pdfs(project_id, location, processor_id, folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            base_name = os.path.splitext(filename)[0]
            output_json = os.path.join(folder_path, base_name + ".json")

            if os.path.exists(output_json):
                print(f"✅ Skipping {filename} — JSON already exists.")
                continue

            input_pdf = os.path.join(folder_path, filename)
            try:
                print(f"\n📄 Processing file: {filename}")
                result = process_pdf_and_return_result(project_id, location, processor_id, input_pdf)

                if result:
                    with open(output_json, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    print(f"✅ Saved to: {output_json}")
                else:
                    print(f"⚠️ No result for: {filename}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {type(e).__name__} - {e}")


if __name__ == "__main__":
    PROJECT_ID = "935917861333"
    LOCATION = "us"
    PROCESSOR_ID = "6078b90020675fba"
    FOLDER_PATH = r"D:\pdf_chtbot\ai-assistant\women_data_pdf"

    process_folder_of_pdfs(PROJECT_ID, LOCATION, PROCESSOR_ID, FOLDER_PATH)

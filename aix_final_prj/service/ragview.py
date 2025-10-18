from django.shortcuts import render
from django.views import View
from .rag_engine import (
    extract_text_from_pdf,
    chunk_text,
    build_vector_db,
    search_similar_chunks,
    generate_answer,
)
import os

class RagView(View):
    template_name = "ragview.html"

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        query = request.POST.get("query")
        context = {}

        # 1. PDF 업로드 처리
        if request.FILES.getlist("pdfs"):
            uploaded_files = request.FILES.getlist("pdfs")
            all_text = ""

            save_dir = os.path.join("static", "uploaded_pdfs")
            os.makedirs(save_dir, exist_ok=True)

            for pdf_file in uploaded_files[:15]:  # 최대 3개
                path = os.path.join(save_dir, pdf_file.name)
                with open(path, "wb") as f:
                    for chunk in pdf_file.chunks():
                        f.write(chunk)
                all_text += extract_text_from_pdf(path)

            chunks = chunk_text(all_text)
            build_vector_db(chunks)

        # 2. 질문 입력 시 검색 + 답변 생성
        if query:
            similar_chunks = search_similar_chunks(query)
            answer = generate_answer(query, similar_chunks)
            context = {
                "query": query,
                "answer": answer
            }
        uploaded_dir = os.path.join("static", "uploaded_pdfs")
        uploaded_filenames = os.listdir(uploaded_dir) if os.path.exists(uploaded_dir) else []
        context["uploaded_filenames"] = uploaded_filenames

        return render(request, self.template_name, context)
    

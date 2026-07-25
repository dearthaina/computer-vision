import os
import cv2
import json
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    st.error("Configure a variável DATABASE_URL no seu arquivo .env ou no Render.")
    st.stop()

# engine = create_engine(DATABASE_URL)
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       
    pool_recycle=300,         
    connect_args={
        "sslmode": "require", 
        "connect_timeout": 15 
    }
)

Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class ImageAnalysis(Base):
    __tablename__ = "image_analyses"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255))
    detected_classes = Column(Text) 
    characteristics_summary = Column(Text) 
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

@st.cache_resource
def load_cv_model():
    return YOLO('yolov8n-seg.pt')

def process_image(image_bytes):
    model = load_cv_model()
    
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = model(img_rgb)
    result = results[0]
    
    annotated_img = result.plot()
    
    detected_objects = []
    class_counts = {}
    
    if result.boxes:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])
            
            detected_objects.append({"objeto": class_name, "confianca": round(conf, 2)})
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
    summary = f"Total de objetos detectados: {len(detected_objects)}. "
    summary += " | ".join([f"{count}x {name}" for name, count in class_counts.items()])
    
    return annotated_img, detected_objects, summary

def save_to_db(filename, detected_objects, summary):
    session = SessionLocal()
    try:
        new_analysis = ImageAnalysis(
            filename=filename,
            detected_classes=json.dumps(detected_objects, ensure_ascii=False),
            characteristics_summary=summary
        )
        session.add(new_analysis)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        st.error(f"Erro ao salvar no banco: {e}")
        return False
    finally:
        session.close()

st.title("Sistema de Segmentação de Imagens")
st.write("Faça o upload de uma imagem para extrair suas características e segmentar os objetos.")

uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagem Original", use_container_width=True)
    
    if st.button("Processar Imagem"):
        with st.spinner("Processando segmentação e extraindo dados..."):
            
            annotated_img, objects_data, summary = process_image(uploaded_file)
            
            st.image(annotated_img, caption="Imagem Segmentada", use_container_width=True)
            
            st.subheader("Resultados da Análise")
            if objects_data:
                st.success(summary)
                st.json(objects_data)
                
                db_success = save_to_db(uploaded_file.name, objects_data, summary)
                if db_success:
                    st.info("Dados gravados com sucesso no banco Neon.tech!")
            else:
                st.warning("Nenhum objeto reconhecido na imagem.")
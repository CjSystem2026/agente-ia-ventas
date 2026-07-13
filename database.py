import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Asegurar que el directorio de datos existe
db_dir = "/app/data"
if not os.path.exists(db_dir):
    os.makedirs(db_dir, exist_ok=True)

# URL de conexión SQLite (usando la carpeta de volumen persistente)
DATABASE_URL = "sqlite:////app/data/sales_agent.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class BingoConfig(Base):
    __tablename__ = "bingo_config"
    
    id = Column(Integer, primary_key=True, index=True)
    prize_pool = Column(Text, nullable=False)
    ticket_price = Column(Float, nullable=False)
    date_time = Column(String(100), nullable=False)
    payment_details = Column(Text, nullable=False)
    special_offers = Column(Text, nullable=True)
    rules = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Lead(Base):
    __tablename__ = "leads"
    
    id = Column(Integer, primary_key=True, index=True)
    phone = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=True)
    status = Column(String(50), default="Interesado")  # "Interesado", "Compró", "No interesado"
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    messages = relationship("ChatMessage", back_populates="lead", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    lead_phone = Column(String(50), ForeignKey("leads.phone", ondelete="CASCADE"), nullable=False)
    sender = Column(String(20), nullable=False)  # "client" o "bot"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    lead = relationship("Lead", back_populates="messages")

# Crear tablas e inicializar configuración por defecto
def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Verificar si ya existe una configuración
        config = db.query(BingoConfig).first()
        if not config:
            # Insertar valores iniciales por defecto para el Bingo
            default_config = BingoConfig(
                prize_pool="S/. 5,000 en premios totales:\n- S/. 1,000 Apagón Mayor\n- 4 Líneas de S/. 500 cada una\n- S/. 2,000 en premios sorpresas.",
                ticket_price=10.0,
                date_time="Sábado 18 de Julio a las 7:30 PM",
                payment_details="Yape o Plin al número 999 888 777 a nombre de Cristian Pérez. También transferencia BCP Cuenta: 191-998877-0-44 (CCI: 002191199887704456). Favor de enviar la captura de pantalla del pago para registrar tus cartones.",
                special_offers="¡Promoción especial! 1 cartón por S/. 10, o aprovecha nuestra súper oferta de 3 cartones por S/. 25 para triplicar tus opciones de ganar.",
                rules="El juego se transmitirá en vivo mediante Zoom y una transmisión privada en Facebook Live. El sistema de marcado es automático (nuestro software canta los números y valida tu cartón), por lo que no es obligatorio estar conectado para ganar. ¡Te avisamos por WhatsApp si ganas!"
            )
            db.add(default_config)
            db.commit()
    finally:
        db.close()

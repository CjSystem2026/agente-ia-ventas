import os
import logging
import httpx
import base64
import asyncio
from datetime import datetime
from fastapi import FastAPI, Depends, Request, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import init_db, SessionLocal, BingoConfig, Lead, ChatMessage
from agent import generate_agent_response

# Motor RAG (tolerante a fallos)
try:
    import rag as rag_engine
    RAG_AVAILABLE = True
except Exception as e:
    RAG_AVAILABLE = False
    logging.getLogger(__name__).warning(f"RAG no disponible en main.py: {e}")

# Configuración de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar base de datos
init_db()

app = FastAPI(title="Agente de Ventas IA - Dashboard")

# Montar archivos estáticos y plantillas
templates_dir = "/app/templates"
static_dir = "/app/static"
os.makedirs(templates_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)
os.makedirs(os.path.join(static_dir, "css"), exist_ok=True)
os.makedirs(os.path.join(static_dir, "js"), exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Variables de configuración desde entorno
EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL", "http://evolution-api:8080")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY", "super_secure_key_999888")
EVOLUTION_INSTANCE_NAME = os.getenv("EVOLUTION_INSTANCE_NAME", "BingoBot")
BINGO_API_URL = os.getenv("BINGO_API_URL", "http://localhost:3000")

# Dependencia para obtener la sesión de la base de datos
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Modelos Pydantic para validar entradas
class ConfigUpdateSchema(BaseModel):
    prize_pool: str
    ticket_price: float
    date_time: str
    payment_details: str
    special_offers: str = ""
    rules: str = ""

class StatusUpdateSchema(BaseModel):
    status: str

class SimulatorChatSchema(BaseModel):
    phone: str
    name: str
    message: str

class ManualMessageSchema(BaseModel):
    text: str

# ----------------- RUTAS HTML -----------------

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request, db: Session = Depends(get_db)):
    config = db.query(BingoConfig).first()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "config": config,
        "instance_name": EVOLUTION_INSTANCE_NAME,
        "gemini_configured": os.getenv("GEMINI_API_KEY") is not None and os.getenv("GEMINI_API_KEY") != "CAMBIA_ESTO_POR_TU_GEMINI_API_KEY"
    })

# ----------------- CONFIG BINGO -----------------

@app.get("/api/config")
async def get_config(db: Session = Depends(get_db)):
    config = db.query(BingoConfig).first()
    if not config:
        raise HTTPException(status_code=404, detail="Configuración no encontrada")
    return config

@app.post("/api/config")
async def update_config(data: ConfigUpdateSchema, db: Session = Depends(get_db)):
    config = db.query(BingoConfig).first()
    if not config:
        config = BingoConfig()
        db.add(config)
    
    config.prize_pool = data.prize_pool
    config.ticket_price = data.ticket_price
    config.date_time = data.date_time
    config.payment_details = data.payment_details
    config.special_offers = data.special_offers
    config.rules = data.rules
    
    db.commit()
    db.refresh(config)
    return {"status": "success", "config": config}

# ----------------- LEADS & CHATS -----------------

@app.get("/api/leads")
async def get_leads(db: Session = Depends(get_db)):
    leads = db.query(Lead).order_by(Lead.updated_at.desc()).all()
    return leads

@app.post("/api/leads/{lead_id}/status")
async def update_lead_status(lead_id: int, data: StatusUpdateSchema, db: Session = Depends(get_db)):
    lead = db.query(Lead).filter(Lead.id == lead_id).first()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead no encontrado")
    lead.status = data.status
    db.commit()
    return {"status": "success", "lead": lead}

@app.get("/api/leads/{phone}/history")
async def get_chat_history(phone: str, db: Session = Depends(get_db)):
    messages = db.query(ChatMessage)\
        .filter(ChatMessage.lead_phone == phone)\
        .order_by(ChatMessage.timestamp.asc())\
        .all()
    return messages

# ----------------- SIMULADOR DE CHAT WEB -----------------

@app.post("/api/simulator/chat")
async def simulator_chat(data: SimulatorChatSchema, db: Session = Depends(get_db)):
    # 1. Crear o recuperar lead
    lead = db.query(Lead).filter(Lead.phone == data.phone).first()
    if not lead:
        lead = Lead(phone=data.phone, name=data.name, status="Interesado")
        db.add(lead)
        db.commit()
        db.refresh(lead)

    # 2. Guardar mensaje del cliente
    client_msg = ChatMessage(lead_phone=data.phone, sender="client", content=data.message)
    db.add(client_msg)
    db.commit()

    # 3. Generar respuesta con Gemini
    response_text = generate_agent_response(db, data.phone, data.message)

    # 4. Guardar respuesta del bot
    bot_msg = ChatMessage(lead_phone=data.phone, sender="bot", content=response_text)
    db.add(bot_msg)
    
    # 5. Actualizar fecha del lead
    lead.updated_at = datetime.utcnow()
    db.commit()

    return {
        "status": "success",
        "response": response_text,
        "lead_status": lead.status
    }

# ----------------- EVOLUTION API INTERFACE -----------------

async def send_whatsapp_message(phone: str, text: str):
    """Llama a Evolution API para enviar un mensaje real de WhatsApp."""
    url = f"{EVOLUTION_API_URL}/message/sendText/{EVOLUTION_INSTANCE_NAME}"
    headers = {
        "apikey": EVOLUTION_API_KEY,
        "Content-Type": "application/json"
    }
    # Limpiar formato de número (debe ser sin signos ni espacios, ej: 51999888777)
    clean_number = phone.split("@")[0]
    payload = {
        "number": clean_number,
        "text": text,
        "delay": 1200, # Pequeño delay de emulación en ms
        "linkPreview": True
    }
    async with httpx.AsyncClient() as client:
        try:
            logger.info(f"Enviando mensaje a {clean_number} a través de Evolution API...")
            response = await client.post(url, json=payload, headers=headers, timeout=10.0)
            if response.status_code in [200, 201]:
                logger.info("Mensaje enviado exitosamente por Evolution API.")
                return True
            else:
                logger.error(f"Error de Evolution API ({response.status_code}): {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error de red al conectar con Evolution API: {str(e)}")
            return False

async def send_whatsapp_presence(phone: str, presence: str = "composing"):
    """Envía un estado de presencia (como 'composing' para escribiendo) al cliente."""
    url = f"{EVOLUTION_API_URL}/chat/sendPresence/{EVOLUTION_INSTANCE_NAME}"
    headers = {
        "apikey": EVOLUTION_API_KEY,
        "Content-Type": "application/json"
    }
    clean_number = phone.split("@")[0]
    payload = {
        "number": clean_number,
        "presence": presence
    }
    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, json=payload, headers=headers, timeout=5.0)
        except Exception as e:
            logger.error(f"Error al enviar presencia a Evolution API: {str(e)}")

@app.post("/api/leads/{phone}/message")
async def send_manual_message(phone: str, data: ManualMessageSchema, db: Session = Depends(get_db)):
    """Envía un mensaje manual del administrador al cliente por WhatsApp y lo registra."""
    jid = phone
    if "@" not in jid:
        clean_num = "".join(filter(str.isdigit, jid))
        if len(clean_num) == 9:
            jid = f"51{clean_num}@s.whatsapp.net"
        else:
            jid = f"{clean_num}@s.whatsapp.net"

    lead = db.query(Lead).filter(Lead.phone == jid).first()
    if not lead:
        lead = Lead(phone=jid, name="Cliente de Web", status="Interesado")
        db.add(lead)
        db.commit()
        db.refresh(lead)

    # Intentar enviar por WhatsApp real
    sent = await send_whatsapp_message(jid, data.text)

    # Registrar el mensaje en base de datos
    admin_msg = ChatMessage(lead_phone=jid, sender="bot", content=data.text)
    db.add(admin_msg)
    
    # Actualizar estado del lead a "Compró" si recibe cartilla
    text_lower = data.text.lower()
    if "cartilla" in text_lower or "jugar" in text_lower or "t=" in text_lower:
        lead.status = "Compró"

    lead.updated_at = datetime.utcnow()
    db.commit()

    return {"status": "success", "sent": sent}

@app.get("/api/evolution/contacts")
async def get_evolution_contacts(only_chats: bool = False):
    """
    Obtiene números de teléfono desde Evolution API.
    - only_chats=False (default): Trae todos los contactos de la agenda.
    - only_chats=True: Trae solo los chats/conversaciones activas (personas con las que ya has hablado).
    """
    headers = {"apikey": EVOLUTION_API_KEY, "Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        try:
            if only_chats:
                # --- Modo: Solo chats activos (conversaciones previas) ---
                logger.info("Obteniendo chats activos desde Evolution API...")
                url = f"{EVOLUTION_API_URL}/chat/findChats/{EVOLUTION_INSTANCE_NAME}"
                response = await client.post(url, json={}, headers=headers, timeout=25.0)
                if response.status_code != 200:
                    return {"status": "error", "message": f"Evolution API retornó {response.status_code}"}
                
                chats_data = response.json()
                clean_numbers = []
                if isinstance(chats_data, list):
                    for chat in chats_data:
                        jid = chat.get("remoteJid", "")
                        # Solo chats individuales con número real de WhatsApp (no grupos ni LIDs internos)
                        if jid and "@s.whatsapp.net" in jid:
                            number = jid.split("@")[0]
                            if number.isdigit():
                                clean_numbers.append(number)
            else:
                # --- Modo: Todos los contactos de la agenda ---
                logger.info("Obteniendo contactos de agenda desde Evolution API...")
                url = f"{EVOLUTION_API_URL}/chat/findContacts/{EVOLUTION_INSTANCE_NAME}"
                response = await client.post(url, json={}, headers=headers, timeout=25.0)
                if response.status_code != 200:
                    return {"status": "error", "message": f"Evolution API retornó {response.status_code}"}

                contacts_data = response.json()
                clean_numbers = []
                if isinstance(contacts_data, list):
                    for contact in contacts_data:
                        jid = contact.get("remoteJid", "")
                        is_group = contact.get("isGroup", False)
                        if jid and not is_group and "@s.whatsapp.net" in jid:
                            number = jid.split("@")[0]
                            if number.isdigit():
                                clean_numbers.append(number)

            unique_numbers = sorted(list(set(clean_numbers)))
            logger.info(f"Se recuperaron {len(unique_numbers)} números válidos (only_chats={only_chats}).")
            return {"status": "success", "contacts": unique_numbers, "total": len(unique_numbers)}

        except Exception as e:
            logger.error(f"Excepción al buscar contactos: {str(e)}")
            return {"status": "error", "message": str(e)}

# ----------------- BASE DE CONOCIMIENTO RAG -----------------

@app.post("/api/rag/upload")
async def rag_upload_document(title: str = Form(...), content: str = Form(...)):
    """Agrega un documento de texto a la base de conocimiento vectorial."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="Motor RAG no disponible")
    if not title.strip() or not content.strip():
        raise HTTPException(status_code=400, detail="El título y contenido no pueden estar vacíos")
    try:
        result = rag_engine.add_document(title.strip(), content.strip())
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"Error al subir documento RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rag/documents")
async def rag_list_documents():
    """Lista todos los documentos en la base de conocimiento."""
    if not RAG_AVAILABLE:
        return {"status": "error", "documents": [], "message": "RAG no disponible"}
    try:
        docs = rag_engine.list_documents()
        return {"status": "success", "documents": docs, "total": len(docs)}
    except Exception as e:
        return {"status": "error", "documents": [], "message": str(e)}

@app.delete("/api/rag/documents/{doc_id}")
async def rag_delete_document(doc_id: str):
    """Elimina un documento de la base de conocimiento por su ID."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="Motor RAG no disponible")
    success = rag_engine.delete_document(doc_id)
    if success:
        return {"status": "success", "message": f"Documento {doc_id} eliminado"}
    raise HTTPException(status_code=500, detail="Error al eliminar el documento")

@app.post("/api/rag/search")
async def rag_search(query: str = Form(...)):
    """Prueba la búsqueda RAG con una consulta (endpoint de depuración)."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="Motor RAG no disponible")
    try:
        context = rag_engine.search_knowledge(query, n_results=3)
        return {"status": "success", "query": query, "context": context or "(Sin resultados relevantes)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evolution/status")
async def check_evolution_status():
    """Consulta el estado de la instancia en Evolution API."""
    url = f"{EVOLUTION_API_URL}/instance/connectionState/{EVOLUTION_INSTANCE_NAME}"
    headers = {"apikey": EVOLUTION_API_KEY}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                # Si la instancia no existe
                return {"instance": {"state": "NOT_INITIALIZED"}}
        except Exception as e:
            return {"instance": {"state": "OFFLINE", "error": str(e)}}

@app.post("/api/evolution/connect")
async def connect_evolution():
    """Crea la instancia en la Evolution API si no existe, o retorna el estado para iniciar el QR."""
    # 1. Intentar crear la instancia
    create_url = f"{EVOLUTION_API_URL}/instance/create"
    headers = {
        "apikey": EVOLUTION_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "instanceName": EVOLUTION_INSTANCE_NAME,
        "token": EVOLUTION_API_KEY,
        "qrcode": True,
        "integration": "WHATSAPP-BAILEYS"
    }
    async with httpx.AsyncClient() as client:
        try:
            # Crear
            create_resp = await client.post(create_url, json=payload, headers=headers, timeout=10.0)
            logger.info(f"Creación de instancia: {create_resp.status_code}")
        except Exception as e:
            logger.error(f"Error creando instancia: {str(e)}")

        # 2. Conectar / Obtener QR
        connect_url = f"{EVOLUTION_API_URL}/instance/connect/{EVOLUTION_INSTANCE_NAME}"
        try:
            connect_resp = await client.get(connect_url, headers=headers, timeout=10.0)
            if connect_resp.status_code == 200:
                return connect_resp.json()
            return {"status": "error", "message": f"Evolution API retornó {connect_resp.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

# ----------------- WEBHOOK DE EVOLUTION API -----------------

async def process_webhook_message(payload: dict, db: Session):
    """Procesa el mensaje recibido de WhatsApp de forma asíncrona."""
    event = payload.get("event")
    if event != "messages.upsert":
        return

    data = payload.get("data", {})
    key = data.get("key", {})
    from_me = key.get("fromMe", False)

    # Ignorar si es enviado por nosotros mismos
    if from_me:
        return

    remote_jid = key.get("remoteJid")
    if not remote_jid or "@g.us" in remote_jid:  # Ignorar grupos
        return

    # Extraer texto del mensaje
    message_obj = data.get("message", {})
    text = ""
    is_image = "imageMessage" in message_obj
    
    if "conversation" in message_obj:
        text = message_obj["conversation"]
    elif "extendedTextMessage" in message_obj:
        text = message_obj["extendedTextMessage"].get("text", "")
    elif is_image:
        text = message_obj["imageMessage"].get("caption", "") or ""

    # Si no es imagen y no hay texto, ignorar
    if not is_image and not text.strip():
        return

    # Extraer nombre del cliente
    push_name = data.get("pushName", "Cliente de WhatsApp")

    # 1. Crear o recuperar Lead
    lead = db.query(Lead).filter(Lead.phone == remote_jid).first()
    if not lead:
        lead = Lead(phone=remote_jid, name=push_name, status="Interesado")
        db.add(lead)
        db.commit()
        db.refresh(lead)
    else:
        # Actualizar nombre si era nulo
        if not lead.name or lead.name == "Cliente de WhatsApp":
            lead.name = push_name

    # 2. Guardar mensaje del cliente en SQLite
    msg_content = text if text.strip() else "[Imagen / Captura de Pago]"
    client_msg = ChatMessage(lead_phone=remote_jid, sender="client", content=msg_content)
    db.add(client_msg)
    db.commit()

    response_text = ""
    is_screenshot_processed = False

    # 3. Si es una imagen, asumir que es la captura de pantalla de pago
    if is_image:
        try:
            img_msg = message_obj.get("imageMessage", {})
            logger.info(f"Campos disponibles en imageMessage: {list(img_msg.keys())}")
            image_b64 = img_msg.get("base64")
            
            if not image_b64:
                message_id = key.get("id")
                if message_id:
                    logger.info(f"base64 ausente en webhook. Descargando media de Evolution API para mensaje ID: {message_id}...")
                    download_url = f"{EVOLUTION_API_URL}/chat/getBase64FromMediaMessage/{EVOLUTION_INSTANCE_NAME}"
                    headers = {
                        "apikey": EVOLUTION_API_KEY,
                        "Content-Type": "application/json"
                    }
                    body = {
                        "message": {
                            "key": {
                                "id": message_id
                            }
                        },
                        "convertToMp4": False
                    }
                    async with httpx.AsyncClient() as client:
                        down_resp = await client.post(download_url, json=body, headers=headers, timeout=20.0)
                        if down_resp.status_code in [200, 201]:
                            resp_json = down_resp.json()
                            image_b64 = resp_json.get("base64")
                            if not image_b64 and "data" in resp_json:
                                image_b64 = resp_json.get("data", {}).get("base64")
                            
                            if image_b64:
                                logger.info("Media descargada exitosamente en formato base64.")
                                if "," in image_b64:
                                    image_b64 = image_b64.split(",")[1]
                            else:
                                logger.warning(f"Respuesta de descarga no contenía campo base64: {resp_json}")
                        else:
                            logger.error(f"Error al llamar a getBase64FromMediaMessage: {down_resp.status_code} - {down_resp.text}")

            if image_b64:
                image_data = base64.b64decode(image_b64)
                
                # Obtener celular limpio de 9 dígitos para el Bingo
                clean_phone = "".join(filter(str.isdigit, remote_jid.split("@")[0]))
                if len(clean_phone) >= 9:
                    clean_phone = clean_phone[-9:]
                
                logger.info(f"Enviando captura de pantalla de {clean_phone} al Bingo en {BINGO_API_URL}...")
                
                async with httpx.AsyncClient() as client:
                    files = {'screenshot': ('screenshot.jpg', image_data, 'image/jpeg')}
                    payload_data = {
                        'phone': clean_phone,
                        'isTrial': 'false',
                        'playerName': push_name
                    }
                    resp = await client.post(f"{BINGO_API_URL}/api/validate-payment", data=payload_data, files=files, timeout=20.0)
                    
                    if resp.status_code == 200:
                        logger.info(f"Captura registrada exitosamente en el Bingo para {clean_phone}.")
                        response_text = "¡Muchas gracias! Recibí tu comprobante. En unos momentos el administrador lo validará y te llegará tu link de juego por aquí. 🎟️"
                        lead.status = "Interesado/Listo para pagar"
                    else:
                        logger.error(f"Bingo API respondió con error {resp.status_code}: {resp.text}")
                        response_text = "¡Hola! Recibí tu imagen, pero hubo un inconveniente al guardarla en el sistema. Asegúrate de enviarla de nuevo o contacta al soporte técnico."
                is_screenshot_processed = True
            else:
                logger.warning("Webhook de imagen no contiene el campo base64.")
        except Exception as e:
            logger.error(f"Error al enviar la imagen al Bingo: {str(e)}")
            response_text = "¡Hola! Recibí tu captura, pero tuvimos un problema técnico temporal al enviarla a nuestro de Bingo. El administrador la revisará manualmente. ¡Gracias!"
            is_screenshot_processed = True

    # 4. Si no fue un comprobante procesado, llamar a Gemini
    if not is_screenshot_processed:
        response_text = generate_agent_response(db, remote_jid, text)

    # 5. Guardar respuesta del bot en SQLite
    bot_msg = ChatMessage(lead_phone=remote_jid, sender="bot", content=response_text)
    db.add(bot_msg)
    
    # Actualizar fecha de modificación del lead
    lead.updated_at = datetime.utcnow()
    db.commit()

    # ---- EMULACIÓN HUMANA ----
    # 1. Enviar estado "Escribiendo..." para simular naturalidad
    await send_whatsapp_presence(remote_jid, "composing")
    
    # 2. Calcular un retraso basado en la longitud del texto (ej: 0.04 seg por carácter)
    # Entre un mínimo de 2.5 y un máximo de 7.5 segundos
    delay_seconds = min(max(len(response_text) * 0.04, 2.5), 7.5)
    logger.info(f"Simulando escritura humana por {delay_seconds:.2f} segundos para {remote_jid}...")
    await asyncio.sleep(delay_seconds)

    # 6. Enviar mensaje de vuelta por WhatsApp
    await send_whatsapp_message(remote_jid, response_text)

@app.post("/webhook/evolution")
async def evolution_webhook(request: Request, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Webhook global que recibe Evolution API para notificarnos de mensajes nuevos."""
    payload = await request.json()
    logger.info(f"Webhook recibido de Evolution API. Evento: {payload.get('event')}")
    
    # Procesar en segundo plano para responder rápido a la API
    background_tasks.add_task(process_webhook_message, payload, db)
    
    return {"status": "received"}

import os
import logging
import google.generativeai as genai
from sqlalchemy.orm import Session
from database import BingoConfig, ChatMessage, Lead

# Configurar logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar API de Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and GEMINI_API_KEY != "CAMBIA_ESTO_POR_TU_GEMINI_API_KEY":
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("API de Gemini configurada exitosamente.")
else:
    logger.warning("ATENCIÓN: GEMINI_API_KEY no está configurada o tiene el valor por defecto. La IA no funcionará.")

# Importar el motor RAG (con tolerancia a fallos si ChromaDB no está listo)
try:
    import rag as rag_engine
    RAG_ENABLED = True
    logger.info("Motor RAG inicializado correctamente.")
except Exception as e:
    RAG_ENABLED = False
    logger.warning(f"Motor RAG no disponible, se usará solo el prompt estático: {e}")

def get_system_instruction(db: Session, rag_context: str = "") -> str:
    """Genera las instrucciones de comportamiento de la IA basadas en la configuración del Bingo."""
    config = db.query(BingoConfig).first()
    if not config:
        return "Eres un asistente de ventas para un Bingo."
    
    special_offers_text = f"\n- OFERTAS ESPECIALES: {config.special_offers}" if config.special_offers else ""
    rules_text = f"\n- REGLAS DEL JUEGO: {config.rules}" if config.rules else ""

    # Bloque de contexto RAG: solo se inyecta si hay información relevante
    rag_block = ""
    if rag_context:
        rag_block = f"""

INFORMACIÓN ADICIONAL DE LA BASE DE CONOCIMIENTO (úsala si es relevante para responder):
---
{rag_context}
---"""
    
    instruction = f"""Eres Sofía, la coordinadora de ventas y animadora oficial del Gran Bingo Vecinal. Tu objetivo principal es convencer de manera entusiasta y amigable a las personas para que compren cartones para el juego.

DATOS OFICIALES DEL BINGO (Usa solo esta información, no inventes nada):
- PREMIOS EN JUEGO: {config.prize_pool}
- COSTO POR CARTÓN: S/. {config.ticket_price}
- FECHA Y HORA DEL EVENTO: {config.date_time}
- MÉTODOS DE PAGO: {config.payment_details}{special_offers_text}{rules_text}{rag_block}

PAUTAS DE COMPORTAMIENTO Y PERSUASIÓN:
1. PERSONALIDAD: Eres muy entusiasta, carismática, alegre y empática. Usa emojis de forma moderada (🎉, 🎰, 🎟️, BINGO!) para mantener la conversación viva y amigable. Habla en español de Latinoamérica.
2. CONVERSACIÓN NATURAL: Mantén tus respuestas relativamente cortas y conversacionales, ideales para leer en WhatsApp. Evita enviar "bloques de texto" gigantes. Es mejor chatear de ida y vuelta.
3. ESTRATEGIA DE VENTAS:
   - Saluda de manera cálida y pregúntales si ya se enteraron del gran Bingo de este sábado.
   - Describe los premios con emoción. ¡El premio mayor es increíble!
   - Si muestran dudas, resalta que el marcado es automático (no tienen que estar conectados a la transmisión de Zoom/FB para ganar) y que es una excelente oportunidad para divertirse en familia y apoyar a la comunidad.
   - Si no responden o se muestran dudosos, usa técnicas de escasez y urgencia con sutileza (ej. "¡Ya nos quedan pocos cartones para esta ronda y varios vecinos se están anotando!").
4. CIERRE DE VENTA:
   - Cuando el cliente decida comprar, guíalo paso a paso. Pregúntale cuántos cartones desea (recuérdale la oferta si aplica) y dile que para completar la compra debe realizar el pago al número indicado y enviarte la captura de pantalla por este mismo chat.
5. BASE DE CONOCIMIENTO: Si en la sección de "INFORMACIÓN ADICIONAL" encuentras datos relevantes para la pregunta del cliente (videos, reglas detalladas, historiales, etc.), úsalos de forma natural en tu respuesta.
6. NO INVENTAR: Si te preguntan algo que no está en los datos oficiales ni en la base de conocimiento, responde amablemente que lo consultarás con la comisión organizadora y vuelve a enfocar la conversación en los cartones.
"""
    return instruction

def generate_agent_response(db: Session, lead_phone: str, new_message: str) -> str:
    """Obtiene el historial de chat, busca contexto en RAG, y llama a Gemini para generar la respuesta."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "CAMBIA_ESTO_POR_TU_GEMINI_API_KEY":
        return "¡Hola! Lo siento, mi cerebro de IA no está configurado correctamente en este momento (falta la API Key de Gemini). Por favor, contacta al administrador del sistema."

    # --- PASO 1: Buscar contexto relevante en la base de conocimiento (RAG) ---
    rag_context = ""
    if RAG_ENABLED:
        try:
            rag_context = rag_engine.search_knowledge(new_message, n_results=3)
            if rag_context:
                logger.info(f"RAG encontró contexto relevante para: '{new_message[:60]}...'")
            else:
                logger.info("RAG: No se encontró contexto relevante, usando solo prompt estático.")
        except Exception as e:
            logger.warning(f"Error al consultar RAG, continuando sin contexto adicional: {e}")

    # --- PASO 2: Obtener historial de los últimos 15 mensajes del lead ---
    history_messages = db.query(ChatMessage)\
        .filter(ChatMessage.lead_phone == lead_phone)\
        .order_by(ChatMessage.timestamp.asc())\
        .limit(15)\
        .all()

    # Formatear el historial para Gemini
    contents = []
    for msg in history_messages:
        role = "user" if msg.sender == "client" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg.content}]
        })
    
    # Agregar el nuevo mensaje que acaba de llegar
    contents.append({
        "role": "user",
        "parts": [{"text": new_message}]
    })

    # --- PASO 3: Generar instrucciones del sistema con contexto RAG inyectado ---
    system_instruction = get_system_instruction(db, rag_context=rag_context)

    try:
        # Llamar al modelo de Gemini
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=system_instruction
        )
        response = model.generate_content(
            contents,
            generation_config={"temperature": 0.7, "max_output_tokens": 800}
        )
        
        # Analizar automáticamente la intención del cliente
        update_lead_status_by_text(db, lead_phone, new_message)
        
        return response.text
    except Exception as e:
        logger.error(f"Error al llamar a la API de Gemini: {str(e)}")
        return "¡Hola! Disculpa, experimenté una pequeña interferencia. ¿Me podrías repetir lo último que dijiste? ¡Estoy lista para ayudarte!"

def update_lead_status_by_text(db: Session, lead_phone: str, text: str):
    """Analiza de forma simple el texto del cliente para clasificar su estado."""
    text_lower = text.lower()
    lead = db.query(Lead).filter(Lead.phone == lead_phone).first()
    if not lead:
        return

    # Si menciona pago, comprobante, captura, yape, listo
    if any(word in text_lower for word in ["yape", "plin", "pague", "comprobante", "captura", "transferencia", "pagado", "listo"]):
        # Posiblemente compró o está listo para comprar
        if lead.status != "Compró":
            lead.status = "Interesado/Listo para pagar"
            
    # Si rechaza rotundamente
    elif any(word in text_lower for word in ["no gracias", "no me interesa", "no quiero", "no me jodas", "sácame de la lista"]):
        lead.status = "No interesado"
        
    db.commit()

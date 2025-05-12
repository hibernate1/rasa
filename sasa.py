# import logging
# from sanic import Sanic
# from sanic.response import json
# from telegram import Update
# from telegram.ext import Application
# from rasa.core.agent import Agent
# import os

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)

# # Initialize Sanic app
# app = Sanic("Reddot25Bot")

# # Telegram Bot Token
# TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7065473546:AAF-eXb_O1qQdwGG1rlWcXMPB5LVMqAOLwE")
# application = Application.builder().token(TOKEN).build()

# # Load Rasa agent globally
# rasa_agent = None

# @app.before_server_start
# async def load_rasa_model(app, loop):
#     global rasa_agent
#     logging.info("Loading Rasa model...")
#     # DO NOT await here — Agent.load is synchronous
#     rasa_agent = Agent.load("models")  # ← FIXED

# @app.route("/telegram/webhook", methods=["POST"])
# async def telegram_webhook(request):
#     global rasa_agent

#     try:
#         data = request.json
#         update = Update.de_json(data, application.bot)
#         message = update.message

#         if not message or not message.text:
#             return json({"status": "ignored", "reason": "No text message"})

#         user_message = message.text
#         user_id = message.chat_id

#         logging.info(f"Received message from {user_id}: {user_message}")

#         # Handle message with Rasa
#         responses = await rasa_agent.handle_text(user_message, sender_id=str(user_id))

#         # Send first response back to user
#         if responses:
#             await message.reply_text(responses[0].get("text", ""))
#         else:
#             await message.reply_text("Sorry, I didn't understand that.")

#         return json({"status": "success"})

#     except Exception as e:
#         logging.error(f"Error handling Telegram update: {e}")
#         return json({"status": "error", "message": str(e)})

# if __name__ == "__main__":
#     try:
#         app.run(host="0.0.0.0", port=5005, debug=True)
#     except Exception as e:
#         logging.error(f"Failed to start server: {e}")

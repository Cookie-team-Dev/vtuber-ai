import asyncio
import discord

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Minimal bot: online as {client.user}")

@client.event
async def on_message(message):
    print(f"Minimal bot: received message: {message.content}")

async def main():
    token = open("/home/cookie/.config/kookie/discord_token").read().strip()
    try:
        await client.start(token)
    except Exception as e:
        print(f"Minimal bot error: {e}")
    finally:
        await client.close()

asyncio.run(main())

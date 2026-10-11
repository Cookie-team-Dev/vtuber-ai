import discord
import asyncio
import signal
import os

TOKEN_FILE = os.path.expanduser("~/.config/kookie/discord_token")

def read_token():
    try:
        with open(TOKEN_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Test bot: online as {client.user}")
    print("Test bot: staying online...")

async def main():
    token = read_token()
    if not token:
        print("No token found")
        return
    
    try:
        await client.start(token)
    except Exception as e:
        print(f"Test bot error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

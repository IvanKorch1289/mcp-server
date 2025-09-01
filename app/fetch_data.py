import aiohttp
import xmltodict
import asyncio
from typing import Dict, Any
from app.settings import settings
from app.utils import cache_response
from app.utils import clean_xml_dict
import logging


logger = logging.getLogger(__name__)


@cache_response(ttl=7200)
async def fetch_from_dadata(inn: str) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Token {settings.DADATA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"query": inn}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                settings.dadata_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    return {"error": f"DaData error: {resp.status}"}
                data = await resp.json()
                suggestions = data.get("suggestions", [])
                if not suggestions:
                    return {"error": "No data found in DaData"}
                return {"status": "success", "data": suggestions[0]["data"]}
    except Exception as e:
        return {"error": f"DaData request failed: {str(e)}"}


@cache_response(ttl=3600)
async def fetch_from_infosphere(inn: str) -> Dict[str, Any]:
    xml_body = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Request>
        <UserID>{settings.infosphere_user_id}</UserID>
        <Password>{settings.infosphere_password}</Password>
        <requestType>check</requestType>
        <sources>fssp</sources>
        <timeout>300</timeout>
        <recursive>0</recursive>
        <async>0</async>
        <PersonReq>
            <inn>{inn}</inn>
        </PersonReq>
    </Request>"""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                settings.infosphere_url,
                data=xml_body,
                headers={"Content-Type": "application/xml"},
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    return {"error": f"InfoSphere error: {resp.status}"}
                text = await resp.text()
                raw_data = xmltodict.parse(text)
                cleaned = clean_xml_dict(raw_data)
                return {"status": "success", "data": cleaned}
    except Exception as e:
        return {"error": f"InfoSphere request failed: {str(e)}"}


@cache_response(ttl=3600)
async def fetch_company_info(inn: str) -> Dict[str, Any]:
    logger.info(f"Fetching data for INN: {inn}")
    if not inn.isdigit() or len(inn) not in (10, 12):
        return {"error": "Invalid INN"}

    dadata_task = asyncio.create_task(fetch_from_dadata(inn))
    infosphere_task = asyncio.create_task(fetch_from_infosphere(inn))

    dadata_res, infosphere_res = await asyncio.gather(dadata_task, infosphere_task, return_exceptions=True)

    if isinstance(dadata_res, Exception):
        dadata_res = {"error": str(dadata_res)}
    if isinstance(infosphere_res, Exception):
        infosphere_res = {"error": str(infosphere_res)}

    return {
        "inn": inn,
        "sources": {
            "dadata": dadata_res,
            "infosphere": infosphere_res
        }
    }

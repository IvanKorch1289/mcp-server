import asyncio
import logging
from typing import Any, Dict

import xmltodict

from app.advanced.utils import cache_response, clean_xml_dict
from app.http_tools.http_client import get_http_client
from app.settings import settings

logger = logging.getLogger(__name__)


@cache_response(ttl=7200)
async def fetch_from_dadata(inn: str) -> Dict[str, Any]:
    client = await get_http_client()
    url = f"{settings.dadata_url}"
    headers = {
        "Authorization": f"Token {settings.dadata_api_key}",
        "Content-Type": "application/json",
    }
    payload = {"query": inn}

    try:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            return {"error": f"DaData error: {resp.status_code}"}
        data = resp.json()
        suggestions = data.get("suggestions", [])
        if not suggestions:
            return {"error": "No data found in DaData"}
        return {"status": "success", "data": suggestions[0]["data"]}
    except Exception as e:
        return {"error": f"DaData request failed: {str(e)}"}


@cache_response(ttl=3600)
async def fetch_from_infosphere(inn: str) -> Dict[str, Any]:
    client = await get_http_client()
    url = settings.infosphere_url
    xml_body = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Request>
        <UserID>{settings.infosphere_login}</UserID>
        <Password>{settings.infosphere_password}</Password>
        <requestType>check</requestType>
        <sources>fssp,bankrot,cbr,egrul,fns,fsin,fmsdb,fms,gosuslugi,mvd,pfr,terrorist</sources>
        <timeout>300</timeout>
        <recursive>0</recursive>
        <async>0</async>
        <PersonReq>
            <inn>{inn}</inn>
        </PersonReq>
    </Request>"""

    try:
        resp = await client.post(
            url, content=xml_body, headers={"Content-Type": "application/xml"}
        )
        if resp.status_code != 200:
            return {"error": f"InfoSphere error: {resp.status_code}"}
        raw_data = xmltodict.parse(resp.text)
        cleaned = clean_xml_dict(raw_data.get("Response", {}).get("Source", []))
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

    dadata_res, infosphere_res = await asyncio.gather(
        dadata_task, infosphere_task, return_exceptions=True
    )

    if isinstance(dadata_res, Exception):
        dadata_res = {"error": str(dadata_res)}
    if isinstance(infosphere_res, Exception):
        infosphere_res = {"error": str(infosphere_res)}

    return {
        "inn": inn,
        "sources": {"dadata": dadata_res, "infosphere": infosphere_res},
    }

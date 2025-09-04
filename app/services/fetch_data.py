import asyncio
from typing import Any, Dict

import xmltodict

from app.advanced.logging_client import logger
from app.advanced.utils import cache_response, clean_xml_dict
from app.services.http_client import AsyncHttpClient
from app.settings import settings


@cache_response(ttl=7200)
async def fetch_from_dadata(inn: str) -> Dict[str, Any]:
    client = await AsyncHttpClient.get_instance()
    url = settings.dadata_url
    headers = {
        "Authorization": f"Token {settings.dadata_api_key}",
        "Content-Type": "application/json",
    }
    payload = {"query": inn}

    try:
        resp = await client.request("POST", url, json=payload, headers=headers)
        if resp.status_code != 200:
            logger.warning(
                f"DaData returned {resp.status_code}: {resp.text}", component="dadata"
            )
            return {"error": f"DaData error: {resp.status_code}"}
        data = resp.json()
        suggestions = data.get("suggestions", [])
        if not suggestions:
            return {"error": "No data found in DaData"}
        return {"status": "success", "data": suggestions[0]["data"]}
    except Exception as e:
        logger.exception(f"DaData request failed for INN {inn}", component="dadata")
        return {"error": f"DaData request failed: {str(e)}"}


@cache_response(ttl=3600)
async def fetch_from_infosphere(inn: str) -> Dict[str, Any]:
    client = await AsyncHttpClient.get_instance()
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
        resp = await client.request(
            "POST", url, content=xml_body, headers={"Content-Type": "application/xml"}
        )
        if resp.status_code != 200:
            logger.warning(
                f"InfoSphere returned {resp.status_code}", component="infosphere"
            )
            return {"error": f"InfoSphere error: {resp.status_code}"}
        raw_data = xmltodict.parse(resp.text)
        cleaned = clean_xml_dict(raw_data.get("Response", {}).get("Source", []))
        return {"status": "success", "data": cleaned}
    except Exception as e:
        logger.exception(
            f"InfoSphere request failed for INN {inn}", component="infosphere"
        )
        return {"error": f"InfoSphere request failed: {str(e)}"}


@cache_response(ttl=9600)
async def fetch_from_casebook(inn: str) -> Dict[str, Any]:
    client = await AsyncHttpClient.get_instance()
    url = settings.casebook_arbitr_url
    params = {
        "sideInn": inn,
        "size": 100,
        "apikey": settings.casebook_api_key,
        "page": 1,
    }

    try:
        # Используем встроенную пагинацию
        all_cases = await client.fetch_all_pages(url=url, params=params)
        return {"status": "success", "data": all_cases}
    except Exception as e:
        logger.exception(f"Casebook request failed for INN {inn}", component="casebook")
        return {"error": f"Casebook request failed: {str(e)}"}


@cache_response(ttl=9600)
async def fetch_company_info(inn: str) -> Dict[str, Any]:
    logger.info(f"Fetching data for INN: {inn}", component="company_info")

    if not inn.isdigit() or len(inn) not in (10, 12):
        logger.warning(f"Invalid INN format: {inn}", component="company_info")
        return {"error": "Invalid INN"}

    # Параллельные запросы
    dadata_task = asyncio.create_task(fetch_from_dadata(inn))
    infosphere_task = asyncio.create_task(fetch_from_infosphere(inn))
    casebook_task = asyncio.create_task(fetch_from_casebook(inn))

    results = await asyncio.gather(
        dadata_task, infosphere_task, casebook_task, return_exceptions=True
    )

    processed_results = {}
    source_names = ["dadata", "infosphere", "casebook"]
    for name, result in zip(source_names, results, strict=False):
        if isinstance(result, Exception):
            logger.error(
                f"Error fetching from {name}: {result}", component="company_info"
            )
            processed_results[name] = {"error": str(result)}
        else:
            processed_results[name] = result

    return {
        "inn": inn,
        "sources": processed_results,
    }

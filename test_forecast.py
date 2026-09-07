import unittest
from unittest.mock import patch
import httpx
from fastapi import HTTPException
import app

class ForecastTests(unittest.IsolatedAsyncioTestCase):
    async def test_split_endpoints_units_and_time_alignment(self):
        calls = []
        class Client:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def get(self, url, params):
                calls.append((url, params))
                field = params['hourly']
                marine = url == app.MARINE_BASE
                times = ['2026-09-07T00:00', '2026-09-07T01:00'] if marine else ['2026-09-07T01:00', '2026-09-07T02:00']
                return httpx.Response(200, request=httpx.Request('GET', url), json={'hourly': {'time': times, field: [0, 2]}, 'hourly_units': {field: 'm' if marine else 'm/s'}})
        with patch.object(app.httpx, 'AsyncClient', return_value=Client()):
            data = await app.fetch_open_meteo(59, 18, ['wave_height', 'wind_speed_10m'])
        self.assertEqual(data['hourly']['wave_height'], [0, 2, None])
        self.assertEqual(data['hourly']['wind_speed_10m'], [None, 0, 2])
        self.assertEqual(calls[1][1]['wind_speed_unit'], 'ms')
        self.assertEqual(data['data_kind'], 'forecast')

    async def test_reject_invalid_input(self):
        for lat, fields in [(91, None), (float('nan'), None), (59, ['not_a_field'])]:
            with self.assertRaises(HTTPException) as e:
                await app.fetch_open_meteo(lat, 18, fields)
            self.assertEqual(e.exception.status_code, 422)

    async def test_provider_errors_fail_closed(self):
        class Client:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def get(self, url, params):
                return httpx.Response(200, request=httpx.Request('GET', url), json={'hourly': {'time': ['bad'], 'wave_height': []}})
        with patch.object(app.httpx, 'AsyncClient', return_value=Client()):
            with self.assertRaises(HTTPException) as e: await app.fetch_open_meteo(59, 18, ['wave_height'])
        self.assertEqual(e.exception.status_code, 502)

    def test_zero_swell_is_not_replaced(self):
        self.assertEqual(app.score_conditions({'swell_wave_height': 0, 'wave_height': 2}, {}), 0)

if __name__ == '__main__': unittest.main()

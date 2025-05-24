import asyncio
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional
import uuid

from models.monitoring import DetectionResult, Alert

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.alert_channels = ["websocket", "redis_pubsub"]  # Can add email, SMS, etc.
        self.alert_cooldown = 60  # seconds between alerts for same person
        
    async def send_alert(self, monitoring_id: str, detection: DetectionResult):
        """Send alert for a detection"""
        try:
            # Check if we should send alert (cooldown period)
            if await self.is_in_cooldown(monitoring_id, detection.user_id):
                logger.info(f"Alert for user {detection.user_id} is in cooldown period")
                return
            # Create alert
            alert = Alert(
                id=str(uuid.uuid4()),
                monitoring_id=monitoring_id,
                detection_id=f"{detection.frame_number}",
                timestamp=datetime.utcnow(),
                alert_type="known_person",
                person_info={
                    "user_id": detection.user_id,
                    "name": detection.person_name,
                    "confidence": detection.confidence
                },
                confidence=detection.confidence,
                message=f"Known person detected: {detection.person_name} (Confidence: {detection.confidence:.2%})",
                image_url=detection.image_url,
                sent_to=[]
            )
            # Send through different channels
            tasks = []
            for channel in self.alert_channels:
                if channel == "websocket":
                    tasks.append(self.send_websocket_alert(alert))
                elif channel == "redis_pubsub":
                    tasks.append(self.send_redis_alert(alert))
            await asyncio.gather(*tasks)
            # Save alert
            await self.save_alert(alert)
            # Set cooldown
            await self.set_cooldown(monitoring_id, detection.user_id)
            logger.info(f"Alert sent for {detection.person_name} in monitoring {monitoring_id}")
        except Exception as e:
            logger.error(f"Failed to send alert: {str(e)}")
    async def send_websocket_alert(self, alert: Alert):
        """Send alert through WebSocket"""
        try:
            alert_data = {
                "type": "alert",
                "alert_id": alert.id,
                "monitoring_id": alert.monitoring_id,
                "timestamp": alert.timestamp.isoformat(),
                "alert_type": alert.alert_type,
                "message": alert.message,
                "person_info": alert.person_info,
                "confidence": alert.confidence,
                "image_url": alert.image_url
            }
            channel = f"alerts:{alert.monitoring_id}"
            await self.redis_client.publish(channel, json.dumps(alert_data))
            await self.redis_client.publish("alerts:global", json.dumps(alert_data))
            alert.sent_to.append("websocket")
        except Exception as e:
            logger.error(f"Failed to send WebSocket alert: {str(e)}")
    async def send_redis_alert(self, alert: Alert):
        """Save alert to Redis for retrieval"""
        try:
            alerts_key = f"alerts:{alert.monitoring_id}"
            await self.redis_client.lpush(alerts_key, json.dumps({
                "id": alert.id,
                "timestamp": alert.timestamp.isoformat(),
                "alert_type": alert.alert_type,
                "message": alert.message,
                "person_info": alert.person_info,
                "confidence": alert.confidence,
                "image_url": alert.image_url,
                "acknowledged": alert.acknowledged
            }))
            await self.redis_client.ltrim(alerts_key, 0, 99)  # Keep last 100 alerts
            await self.redis_client.expire(alerts_key, 86400)  # 24 hours
            alert.sent_to.append("redis_pubsub")
        except Exception as e:
            logger.error(f"Failed to save alert to Redis: {str(e)}")
    async def save_alert(self, alert: Alert):
        """Save alert details"""
        try:
            alert_key = f"alert:{alert.id}"
            alert_data = {
                "id": alert.id,
                "monitoring_id": alert.monitoring_id,
                "detection_id": alert.detection_id,
                "timestamp": alert.timestamp.isoformat(),
                "alert_type": alert.alert_type,
                "person_info": alert.person_info,
                "confidence": alert.confidence,
                "message": alert.message,
                "image_url": alert.image_url,
                "sent_to": alert.sent_to,
                "acknowledged": alert.acknowledged,
                "acknowledged_by": alert.acknowledged_by,
                "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
            }
            await self.redis_client.setex(
                alert_key,
                86400,  # 24 hours
                json.dumps(alert_data)
            )
        except Exception as e:
            logger.error(f"Failed to save alert: {str(e)}")
    async def is_in_cooldown(self, monitoring_id: str, user_id: str) -> bool:
        """Check if alert for user is in cooldown period"""
        if not user_id:
            return False
        cooldown_key = f"alert_cooldown:{monitoring_id}:{user_id}"
        exists = await self.redis_client.exists(cooldown_key)
        return exists > 0
    async def set_cooldown(self, monitoring_id: str, user_id: str):
        """Set cooldown for user alerts"""
        if not user_id:
            return
        cooldown_key = f"alert_cooldown:{monitoring_id}:{user_id}"
        await self.redis_client.setex(cooldown_key, self.alert_cooldown, "1")
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        try:
            alert_key = f"alert:{alert_id}"
            alert_data = await self.redis_client.get(alert_key)
            if alert_data:
                alert = json.loads(alert_data)
                alert["acknowledged"] = True
                alert["acknowledged_by"] = acknowledged_by
                alert["acknowledged_at"] = datetime.utcnow().isoformat()
                await self.redis_client.setex(
                    alert_key,
                    86400,  # 24 hours
                    json.dumps(alert)
                )
                ack_data = {
                    "type": "alert_acknowledged",
                    "alert_id": alert_id,
                    "acknowledged_by": acknowledged_by,
                    "acknowledged_at": alert["acknowledged_at"]
                }
                await self.redis_client.publish(
                    f"alerts:{alert['monitoring_id']}",
                    json.dumps(ack_data)
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {str(e)}")
            return False
    async def get_alerts(
        self,
        monitoring_id: str,
        limit: int = 50,
        include_acknowledged: bool = True
    ) -> List[Dict]:
        """Get alerts for a monitoring session"""
        try:
            alerts_key = f"alerts:{monitoring_id}"
            alerts_data = await self.redis_client.lrange(alerts_key, 0, limit - 1)
            alerts = []
            for alert_json in alerts_data:
                alert = json.loads(alert_json)
                if include_acknowledged or not alert.get("acknowledged", False):
                    alerts.append(alert)
            return alerts
        except Exception as e:
            logger.error(f"Failed to get alerts: {str(e)}")
            return []
    async def get_alert_stats(self, monitoring_id: str) -> Dict:
        """Get alert statistics for a monitoring session"""
        try:
            alerts = await self.get_alerts(monitoring_id, limit=1000)
            total_alerts = len(alerts)
            acknowledged = sum(1 for a in alerts if a.get("acknowledged", False))
            pending = total_alerts - acknowledged
            by_person = {}
            for alert in alerts:
                person_id = alert.get("person_info", {}).get("user_id", "unknown")
                if person_id not in by_person:
                    by_person[person_id] = 0
                by_person[person_id] += 1
            return {
                "total_alerts": total_alerts,
                "acknowledged": acknowledged,
                "pending": pending,
                "by_person": by_person,
                "last_alert_time": alerts[0]["timestamp"] if alerts else None
            }
        except Exception as e:
            logger.error(f"Failed to get alert stats: {str(e)}")
            return {}

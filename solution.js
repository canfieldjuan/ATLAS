commit-message: feat: add marketer-facing verification MCP server for content operating model

description: This change introduces a new MCP server that exposes the content-ops verification workflow to non-coding marketers. The server provides tools for structured brief creation, asset schema validation, and content verification. It includes:

- A new `verification_server.py` module that implements the MCP server with three main tools:
  - `create_content_brief`: Guides marketers through creating structured content briefs
  - `validate_asset_schema`: Validates content assets against the required schema
  - `verify_content`: Runs the full verification workflow on submitted content

- Integration with the existing content operating model workflow (referenced in issue #1338)
- Proper error handling and input validation
- Clear documentation for marketers on how to use each tool

The server is designed to be used with Claude Desktop or ChatGPT connectors, making the verification workflow accessible to non-technical team members.

```verification_server.py
"""
Marketer-facing Verification MCP Server

This MCP server exposes the content-ops verification workflow to non-coding marketers.
It provides tools for structured brief creation, asset schema validation, and content verification.

Usage:
    python verification_server.py
    
    Or integrate with Claude Desktop via MCP configuration.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field

# MCP SDK imports
try:
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationOptions
    import mcp.server.stdio
    import mcp.types as types
except ImportError:
    print("Error: MCP SDK not installed. Install with: pip install mcp")
    sys.exit(1)


@dataclass
class ContentBrief:
    """Structured content brief for marketers."""
    title: str
    target_audience: str
    content_type: str  # blog_post, social_media, email, landing_page, etc.
    key_message: str
    tone_of_voice: str
    target_keywords: List[str] = field(default_factory=list)
    call_to_action: str = ""
    word_count_target: Optional[int] = None
    additional_requirements: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "draft"


@dataclass
class AssetSchema:
    """Schema definition for content assets."""
    asset_type: str
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, str] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of content verification."""
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# Predefined asset schemas for common content types
ASSET_SCHEMAS = {
    "blog_post": AssetSchema(
        asset_type="blog_post",
        required_fields=["title", "body", "meta_description", "slug"],
        optional_fields=["featured_image", "tags", "category", "author_bio"],
        field_types={
            "title": "string",
            "body": "string",
            "meta_description": "string",
            "slug": "string",
            "featured_image": "url",
            "tags": "array",
            "category": "string",
            "author_bio": "string"
        },
        validation_rules={
            "title_max_length": 120,
            "meta_description_max_length": 160,
            "body_min_length": 300,
            "slug_pattern": "^[a-z0-9-]+$"
        }
    ),
    "social_media": AssetSchema(
        asset_type="social_media",
        required_fields=["platform", "content", "post_type"],
        optional_fields=["media_urls", "hashtags", "link", "call_to_action"],
        field_types={
            "platform": "string",
            "content": "string",
            "post_type": "string",
            "media_urls": "array",
            "hashtags": "array",
            "link": "url",
            "call_to_action": "string"
        },
        validation_rules={
            "content_max_length": 280,
            "hashtags_max_count": 5
        }
    ),
    "email": AssetSchema(
        asset_type="email",
        required_fields=["subject_line", "preheader", "body", "sender_name"],
        optional_fields=["recipient_list", "personalization_fields", "attachments"],
        field_types={
            "subject_line": "string",
            "preheader": "string",
            "body": "string",
            "sender_name": "string",
            "recipient_list": "array",
            "personalization_fields": "object",
            "attachments": "array"
        },
        validation_rules={
            "subject_line_max_length": 60,
            "preheader_max_length": 100,
            "body_min_length": 100
        }
    ),
    "landing_page": AssetSchema(
        asset_type="landing_page",
        required_fields=["headline", "subheadline", "body", "cta_text", "cta_url"],
        optional_fields=["hero_image", "testimonials", "features_list", "form_fields"],
        field_types={
            "headline": "string",
            "subheadline": "string",
            "body": "string",
            "cta_text": "string",
            "cta_url": "url",
            "hero_image": "url",
            "testimonials": "array",
            "features_list": "array",
            "form_fields": "array"
        },
        validation_rules={
            "headline_max_length": 100,
            "cta_text_max_length": 50,
            "body_min_length": 200
        }
    )
}


class VerificationServer:
    """MCP server for content verification workflow."""
    
    def __init__(self):
        self.server = Server("content-verification-server")
        self.setup_tools()
        self.setup_resources()
        
    def setup_tools(self):
        """Register MCP tools."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="create_content_brief",
                    description="Create a structured content brief for your content piece. "
                                "This guides you through specifying all required elements.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Title of the content piece"
                            },
                            "target_audience": {
                                "type": "string",
                                "description": "Who is this content for?"
                            },
                            "content_type": {
                                "type": "string",
                                "enum": ["blog_post", "social_media", "email", "landing_page"],
                                "description": "Type of content you're creating"
                            },
                            "key_message": {
                                "type": "string",
                                "description": "The main message you want to convey"
                            },
                            "tone_of_voice": {
                                "type": "string",
                                "description": "Tone: professional, casual, humorous, etc."
                            },
                            "target_keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Keywords to target (optional)"
                            },
                            "call_to_action": {
                                "type": "string",
                                "description": "What should the reader do? (optional)"
                            },
                            "word_count_target": {
                                "type": "integer",
                                "description": "Target word count (optional)"
                            },
                            "additional_requirements": {
                                "type": "string",
                                "description": "Any other requirements (optional)"
                            }
                        },
                        "required": ["title", "target_audience", "content_type", "key_message", "tone_of_voice"]
                    }
                ),
                types.Tool(
                    name="validate_asset_schema",
                    description="Validate your content asset against the required schema. "
                                "Checks if all required fields are present and correctly formatted.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "asset_type": {
                                "type": "string",
                                "enum": ["blog_post", "social_media", "email", "landing_page"],
                                "description": "Type of content asset"
                            },
                            "asset_data": {
                                "type": "object",
                                "description": "The content asset data to validate"
                            }
                        },
                        "required": ["asset_type", "asset_data"]
                    }
                ),
                types.Tool(
                    name="verify_content",
                    description="Run the full verification workflow on submitted content. "
                                "This checks schema compliance, brand guidelines, and quality standards.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content_brief": {
                                "type": "object",
                                "description": "The content brief (from create_content_brief)"
                            },
                            "asset_data": {
                                "type": "object",
                                "description": "The actual content asset to verify"
                            },
                            "asset_type": {
                                "type": "string",
                                "enum": ["blog_post", "social_media", "email", "landing_page"],
                                "description": "Type of content asset"
                            }
                        },
                        "required": ["content_brief", "asset_data", "asset_type"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: Dict[str, Any]
        ) -> List[types.TextContent]:
            if name == "create_content_brief":
                result = self.create_content_brief(arguments)
            elif name == "validate_asset_schema":
                result = self.validate_asset_schema(arguments)
            elif name == "verify_content":
                result = self.verify_content(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    
    def setup_resources(self):
        """Register MCP resources."""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[types.Resource]:
            return [
                types.Resource(
                    uri="content://schemas",
                    name="Content Asset Schemas",
                    description="Available content asset schemas for verification",
                    mimeType="application/json"
                ),
                types.Resource(
                    uri="content://briefs",
                    name="Content Briefs",
                    description="Your saved content briefs",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            if uri == "content://schemas":
                return json.dumps({
                    asset_type: asdict(schema) 
                    for asset_type, schema in ASSET_SCHEMAS.items()
                }, indent=2)
            elif uri == "content://briefs":
                # In a real implementation, this would fetch from a database
                return json.dumps({"briefs": [], "message": "No briefs saved yet"}, indent=2)
            else:
                raise ValueError(f"Unknown resource: {uri}")
    
    def create_content_brief(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured content brief."""
        try:
            brief = ContentBrief(
                title=arguments["title"],
                target_audience=arguments["target_audience"],
                content_type=arguments["content_type"],
                key_message=arguments["key_message"],
                tone_of_voice=arguments["tone_of_voice"],
                target_keywords=arguments.get("target_keywords", []),
                call_to_action=arguments.get("call_to_action", ""),
                word_count_target=arguments.get("word_count_target"),
                additional_requirements=arguments.get("additional_requirements", "")
            )
            
            return {
                "success": True,
                "message": "Content brief created successfully!",
                "brief": asdict(brief),
                "next_steps": [
                    "Review the brief and make any adjustments",
                    "Create your content following the brief",
                    "Use validate_asset_schema to check your content structure",
                    "Use verify_content for final verification"
                ]
            }
        except KeyError as e:
            return {
                "success": False,
                "error": f"Missing required field: {str(e)}",
                "required_fields": ["title", "target_audience", "content_type", "key_message", "tone_of_voice"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create brief: {str(e)}"
            }
    
    def validate_asset_schema(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content asset against schema."""
        try:
            asset_type = arguments["asset_type"]
            asset_data = arguments["asset_data"]
            
            if asset_type not in ASSET_SCHEMAS:
                return {
                    "success": False,
                    "error": f"Unknown asset type: {asset_type}",
                    "available_types": list(ASSET_SCHEMAS.keys())
                }
            
            schema = ASSET_SCHEMAS[asset_type]
            errors = []
            warnings = []
            
            # Check required fields
            for field in schema.required_fields:
                if field not in asset_data or not asset_data[field]:
                    errors.append(f"Missing required field: {field}")
            
            # Check field types
            for field, value in asset_data.items():
                if field in schema.field_types:
                    expected_type = schema.field_types[field]
                    if expected_type == "array" and not isinstance(value, list):
                        errors.append(f"Field '{field}' should be an array")
                    elif expected_type == "url" and not isinstance(value, str):
                        errors.append(f"Field '{field}' should be a URL string")
                    elif expected_type == "object" and not isinstance(value, dict):
                        errors.append(f"Field '{field}' should be an object")
            
            # Check validation rules
            for rule, value in schema.validation_rules.items():
                if rule == "title_max_length" and "title" in asset_data:
                    if len(asset_data["title"]) > value:
                        warnings.append(f"Title exceeds {value} characters")
                elif rule == "meta_description_max_length" and "meta_description" in asset_data:
                    if len(asset_data["meta_description"]) > value:
                        warnings.append(f"Meta description exceeds {value} characters")
                elif rule == "body_min_length" and "body" in asset_data:
                    if len(asset_data["body"]) < value:
                        warnings.append(f"Body is too short (minimum {value} characters)")
            
            passed = len(errors) == 0
            
            return {
                "success": True,
                "passed": passed,
                "errors": errors,
                "warnings": warnings,
                "summary": "All checks passed!" if passed else f"Found {len(errors)} error(s) and {len(warnings)} warning(s)"
            }
            
        except KeyError as e:
            return {
                "success": False,
                "error": f"Missing required field: {str(e)}",
                "required_fields": ["asset_type", "asset_data"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Validation failed: {str(e)}"
            }
    
    def verify_content(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Run full verification workflow."""
        try:
            content_brief = arguments["content_brief"]
            asset_data = arguments["asset_data"]
            asset_type = arguments["asset_type"]
            
            # Step 1: Validate schema
            schema_result = self.validate_asset_schema({
                "asset_type": asset_type,
                "asset_data": asset_data
            })
            
            if not schema_result.get("success", False):
                return {
                    "success": False,
                    "error": "Schema validation failed",
                    "details": schema_result
                }
            
            # Step 2: Check brief alignment
            brief_checks = self._check_brief_alignment(content_brief, asset_data)
            
            # Step 3: Quality checks
            quality_checks = self._run_quality_checks(asset_data, asset_type)
            
            # Compile results
            all_checks = {
                "schema_validation": schema_result["passed"],
                "brief_alignment": brief_checks["passed"],
                "quality_standards": quality_checks["passed"]
            }
            
            all_errors = schema_result.get("errors", []) + brief_checks.get("errors", []) + quality_checks.get("errors", [])
            all_warnings = schema_result.get("warnings", []) + brief_checks.get("warnings", []) + quality_checks.get("warnings", [])
            
            verification = VerificationResult(
                passed=all(all_checks.values()),
                checks=all_checks,
                errors=all_errors,
                warnings=all_warnings,
                suggestions=brief_checks.get("suggestions", []) + quality_checks.get("suggestions", [])
            )
            
            return {
                "success": True,
                "verification": asdict(verification),
                "summary": "Content verified successfully!" if verification.passed else "Content needs revisions",
                "next_steps": [
                    "Review any errors and warnings",
                    "Make necessary revisions",
                    "Re-run verification after changes",
                    "Submit for final approval"
                ] if not verification.passed else [
                    "Content is ready for submission",
                    "Proceed with publishing workflow"
                ]
            }
            
        except KeyError as e:
            return {
                "success": False,
                "error": f"Missing required field: {str(e)}",
                "required_fields": ["content_brief", "asset_data", "asset_type"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Verification failed: {str(e)}"
            }
    
    def _check_brief_alignment(self, brief: Dict[str, Any], asset_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if content aligns with the brief."""
        errors = []
        warnings = []
        suggestions = []
        
        # Check title alignment
        if "title" in brief and "title
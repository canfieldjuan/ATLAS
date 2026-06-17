import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { documentService } from '@/lib/graphrag';

export const runtime = 'nodejs';
export const maxDuration = 3600; // 60 minutes for large document processing

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id: documentId } = await params;

    console.log(`[GraphRAG Process] ===== PROCESSING REQUEST RECEIVED =====`);
    console.log(`[GraphRAG Process] Document ID: ${documentId}`);
    console.log(`[GraphRAG Process] Timestamp: ${new Date().toISOString()}`);

    const authHeader = request.headers.get('authorization');
    console.log(`[GraphRAG Process] Has Auth Header: ${!!authHeader}`);

    if (!authHeader) {
      console.error(`[GraphRAG Process] Missing authorization header`);
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
    if (!supabaseUrl || !supabaseAnonKey) {
      console.error('[GraphRAG Process] Supabase environment is not configured');
      return NextResponse.json(
        { error: 'Supabase configuration missing' },
        { status: 500 }
      );
    }
    const supabase = createClient(supabaseUrl, supabaseAnonKey, {
      global: {
        headers: {
          Authorization: authHeader,
        },
      },
    });

    const { data: { user }, error: authError } = await supabase.auth.getUser();

    if (authError || !user) {
      console.error(`[GraphRAG Process] Auth failed:`, authError?.message);
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    console.log(`[GraphRAG Process] ===== AUTHENTICATED =====`);
    console.log(`[GraphRAG Process] User ID: ${user.id}`);
    console.log(`[GraphRAG Process] User Email: ${user.email}`);
    console.log(`[GraphRAG Process] GRAPHITI_API_URL: ${process.env.GRAPHITI_API_URL || 'NOT SET'}`);
    console.log(`[GraphRAG Process] NODE_ENV: ${process.env.NODE_ENV}`);
    console.log(`[GraphRAG Process] Starting document processing...`);

    const result = await documentService.processDocument(supabase, documentId);

    if (result.error) {
      console.error(`[GraphRAG Process] Processing failed for ${documentId}: ${result.error}`);
      return NextResponse.json(
        {
          success: false,
          documentId,
          error: result.error,
        },
        { status: 500 }
      );
    }

    console.log(`[GraphRAG Process] Successfully processed document ${documentId}`);
    console.log(`[GraphRAG Process] Episodes created: ${result.episodeIds.length}`);
    console.log(`[GraphRAG Process] Episode IDs: ${result.episodeIds.join(', ')}`);

    return NextResponse.json({
      success: true,
      documentId,
      processed: result.processed,
      episodeIds: result.episodeIds,
    });
  } catch (error) {
    console.error('[GraphRAG Process] Unexpected error:', error);
    console.error('[GraphRAG Process] Error stack:', error instanceof Error ? error.stack : 'No stack trace');
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to process document',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}

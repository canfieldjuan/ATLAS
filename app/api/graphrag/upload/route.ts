import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { documentService } from '@/lib/graphrag';

export const runtime = 'nodejs';
export const maxDuration = 300; // 5 minutes for document processing

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;

    // Extract optional chunking settings from form data
    const chunkSizeStr = formData.get('chunkSize') as string | null;
    const chunkOverlapStr = formData.get('chunkOverlap') as string | null;
    const maxChunkCharsStr = formData.get('maxChunkChars') as string | null;

    // Parse chunking options (undefined if not provided = use server defaults)
    const chunkSize = chunkSizeStr ? parseInt(chunkSizeStr, 10) : undefined;
    const chunkOverlap = chunkOverlapStr ? parseInt(chunkOverlapStr, 10) : undefined;
    const maxChunkChars = maxChunkCharsStr ? parseInt(maxChunkCharsStr, 10) : undefined;

    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }

    const authHeader = request.headers.get('authorization');
    if (!authHeader) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
    if (!supabaseUrl || !supabaseAnonKey) {
      console.error('[GraphRAG Upload] Supabase environment is not configured');
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
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    // Try to upload file - if duplicate, automatically create new version
    let document;
    let isNewVersion = false;

    try {
      document = await documentService.uploadOnly(supabase, {
        userId: user.id,
        file,
        metadata: {
          uploadedAt: new Date().toISOString(),
        },
      });
    } catch (uploadError) {
      // If document exists, automatically create new version
      if (uploadError instanceof Error && uploadError.message === 'DOCUMENT_EXISTS_USE_UPDATE') {
        console.log('[GraphRAG Upload] Document exists, creating new version automatically');
        document = await documentService.updateDocument(supabase, {
          userId: user.id,
          file,
          metadata: {
            uploadedAt: new Date().toISOString(),
            updatedVia: 'api',
          },
        });
        isNewVersion = true;
      } else {
        throw uploadError;
      }
    }

    // Process document SYNCHRONOUSLY (like promote-to-graph does)
    console.log(`[GraphRAG Upload] ===== STARTING SYNCHRONOUS PROCESSING =====`);
    console.log(`[GraphRAG Upload] Document ID: ${document.id}`);
    console.log(`[GraphRAG Upload] Filename: ${document.filename}`);
    console.log(`[GraphRAG Upload] Already processed: ${document.processed}`);
    console.log(`[GraphRAG Upload] Timestamp: ${new Date().toISOString()}`);

    if (!document.processed) {
      console.log(`[GraphRAG Upload] Processing document synchronously...`);
      if (chunkSize || chunkOverlap || maxChunkChars) {
        console.log(`[GraphRAG Upload] Custom chunking settings:`, { chunkSize, chunkOverlap, maxChunkChars });
      }

      const result = await documentService.processDocument(supabase, document.id, {
        chunkSize,
        chunkOverlap,
        maxChunkChars,
      });

      if (result.error) {
        console.error(`[GraphRAG Upload] Processing failed:`, result.error);
        return NextResponse.json(
          {
            success: false,
            error: 'Failed to process document',
            details: result.error,
          },
          { status: 500 }
        );
      }

      console.log(`[GraphRAG Upload] ===== PROCESSING COMPLETE =====`);
      console.log(`[GraphRAG Upload] Episodes created: ${result.episodeIds.length}`);
      console.log(`[GraphRAG Upload] Episode IDs: ${result.episodeIds.join(', ')}`);

      return NextResponse.json({
        success: true,
        document,
        processed: true,
        episodeIds: result.episodeIds,
        message: isNewVersion
          ? `New version (v${document.version}) created and processed successfully`
          : `File uploaded and processed successfully`,
        isNewVersion,
      });
    } else {
      console.log(`[GraphRAG Upload] Document ${document.id} already processed (version update)`);

      return NextResponse.json({
        success: true,
        document,
        processed: true,
        message: isNewVersion
          ? `New version (v${document.version}) created (already processed)`
          : `File uploaded (already processed)`,
        isNewVersion,
      });
    }
  } catch (error) {
    console.error('[GraphRAG Upload] Error:', error);

    // Handle other errors
    return NextResponse.json(
      {
        error: 'Failed to process file',
        details: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}

# oRPC + React Query + Next.js Complete Setup Guide
# this was generated via AI, caution is advised
# Ai confused trpc and orpc, so some answers are totally wrong
A comprehensive guide to setting up oRPC with React Query in a Next.js project with SSR support, Better Auth integration, and role-based routing templates.

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [oRPC Server Setup](#orpc-server-setup)
4. [React Query Provider Setup](#react-query-provider-setup)
5. [oRPC Client Setup](#orpc-client-setup)
6. [Router Templates](#router-templates)
7. [Middleware for Route Protection](#middleware-for-route-protection)
8. [Usage Examples](#usage-examples)

---

## Installation

Install the required dependencies:

npm install @orpc/core @orpc/react-query @orpc/openapi @orpc/server
npm install @tanstack/react-query
npm install better-auth

---

## Project Structure

Organize your project as follows:

src/
├── app/
│   ├── layout.tsx                 # Root layout with providers
│   ├── providers.tsx              # React Query provider
│   ├── (public)/
│   │   ├── page.tsx
│   │   └── login/page.tsx
│   ├── (private)/
│   │   ├── layout.tsx             # Private layout wrapper
│   │   ├── dashboard/page.tsx
│   │   └── profile/page.tsx
│   ├── (admin)/
│   │   ├── layout.tsx             # Admin layout wrapper
│   │   ├── users/page.tsx
│   │   └── settings/page.tsx
│   └── api/
│       ├── auth/[...all]/route.ts
│       └── orpc/[...path]/route.ts
├── server/
│   ├── auth.ts                    # Better Auth configuration
│   ├── orpc/
│   │   ├── index.ts               # oRPC server instance
│   │   ├── router.ts              # Main router
│   │   ├── public.ts              # Public procedures
│   │   ├── private.ts             # Private procedures
│   │   └── admin.ts               # Admin procedures
│   └── db/
│       └── queries.ts
└── client/
    └── orpc.ts                    # oRPC client setup

---

## oRPC Server Setup

### 1. Server Instance (`server/orpc/index.ts`)

import { os } from '@orpc/core';
import { getAuth } from '@/server/auth';
import { publicRouter } from './public';
import { privateRouter } from './private';
import { adminRouter } from './admin';

// Create base oRPC server instance
export const orpc = os
  .context<{ auth?: Awaited<ReturnType<typeof getAuth>> }>()
  .router({
    public: publicRouter,
    private: privateRouter,
    admin: adminRouter,
  });

export type ORPC = typeof orpc;

### 2. Auth Context Helper

In your server setup, extract auth context from the request. Better Auth handles session validation automatically:

// server/orpc/index.ts (additions)
export async function getContextFromRequest(req: Request) {
  // Better Auth's getAuth() handles session retrieval and validation
  // No database queries here - just extracts session from request
  const auth = await getAuth(req);
  return { auth };
}

### 3. API Route (`app/api/orpc/[...path]/route.ts`)

import { toNextJsHandler } from '@orpc/server/next';
import { orpc, getContextFromRequest } from '@/server/orpc';
import { NextRequest } from 'next/server';

const handler = toNextJsHandler(orpc, {
  getContext: getContextFromRequest,
});

export const GET = handler;
export const POST = handler;

---

## React Query Provider Setup

### 1. Create Provider Component (`app/providers.tsx`)

'use client';

import React from 'react';
import {
  QueryClient,
  QueryClientProvider,
  HydrationBoundary,
} from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

interface ProvidersProps {
  children: React.ReactNode;
  initialState?: unknown;
}

export function Providers({ children, initialState }: ProvidersProps) {
  const [queryClient] = React.useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000, // 1 minute
            gcTime: 5 * 60 * 1000, // 5 minutes (formerly cacheTime)
            retry: 1,
            refetchOnWindowFocus: false,
          },
          mutations: {
            retry: 1,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <HydrationBoundary state={initialState}>
        {children}
      </HydrationBoundary>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

### 2. Get Query Client (`app/get-query-client.ts`)

import { QueryClient } from '@tanstack/react-query';

let client: QueryClient | undefined;

export function getQueryClient() {
  if (!client) {
    client = new QueryClient({
      defaultOptions: {
        queries: {
          staleTime: 60 * 1000,
          gcTime: 5 * 60 * 1000,
          retry: 1,
        },
      },
    });
  }
  return client;
}

### 3. Update Root Layout (`app/layout.tsx`)

import { Providers } from './providers';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Providers initialState={undefined}>
          {children}
        </Providers>
      </body>
    </html>
  );
}

---

## oRPC Client Setup

### 1. Client Instance (`client/orpc.ts`)

'use client';

import { createORPCClient } from '@orpc/react-query';
import { ORPC } from '@/server/orpc';

// Client-side oRPC client with React Query integration
export const orpcClient = createORPCClient<ORPC>({
  baseURL:
    process.env.NEXT_PUBLIC_API_URL || `${typeof window !== 'undefined' ? window.location.origin : ''}/api/orpc`,
  headers: {
    'Content-Type': 'application/json',
  },
});

### 2. Server-Side Client (`client/orpc.server.ts`)

import { os } from '@orpc/core';
import { orpc as serverOrpc } from '@/server/orpc';
import { getAuth } from '@/server/auth';

/**
 * Server-side oRPC client for direct invocation
 * Use this in Server Components or server actions
 */
export async function getServerORPC(request?: Request) {
  const auth = await getAuth(request);

  return serverOrpc.context({ auth });
}

---

## Router Templates

### 1. Public Router (`server/orpc/public.ts`)

import { os } from '@orpc/core';
import { z } from 'zod';

export const publicRouter = os.router({
  // Public procedures - no auth required
  getInfo: os
    .input(z.object({ id: z.string() }))
    .handler(async ({ input }) => {
      // Public data - no auth needed
      return { id: input.id, info: 'Public data' };
    }),

  // Note: Auth endpoints are typically handled via better-auth directly
  // The oRPC procedures here are for domain logic, not authentication
  getPublicStats: os
    .handler(async () => {
      // Return publicly available statistics
      return { users: 0, apiCalls: 0 };
    }),
});

### 2. Private Router (`server/orpc/private.ts`)

import { os } from '@orpc/core';
import { z } from 'zod';
import { TRPCError } from '@trpc/server';

/**
 * Middleware for authenticated requests
 */
function ensureAuth() {
  return os.middleware(({ next, context }) => {
    if (!context.auth?.session) {
      throw new Error('Unauthorized: Authentication required');
    }
    return next({
      context: {
        ...context,
        auth: context.auth,
        userId: context.auth.session.userId,
      },
    });
  });
}

export const privateRouter = os
  .use(ensureAuth())
  .router({
    // User profile
    getProfile: os.handler(async ({ context }) => {
      // Auth check happens in middleware
      return {
        id: context.userId,
        email: context.auth?.session?.user?.email,
        name: context.auth?.session?.user?.name,
      };
    }),

    updateProfile: os
      .input(
        z.object({
          name: z.string().optional(),
          bio: z.string().optional(),
        })
      )
      .handler(async ({ input, context }) => {
        // Procedure logic only - auth already verified
        return { success: true, userId: context.userId };
      }),

    // List user resources
    listResources: os
      .input(
        z.object({
          limit: z.number().default(10),
          offset: z.number().default(0),
        })
      )
      .handler(async ({ input, context }) => {
        // Procedure logic only - auth already verified
        return {
          items: [],
          total: 0,
          limit: input.limit,
          offset: input.offset,
        };
      }),
  });

### 3. Admin Router (`server/orpc/admin.ts`)

import { os } from '@orpc/core';
import { z } from 'zod';

/**
 * Middleware for admin requests
 */
function ensureAdmin() {
  return os.middleware(({ next, context }) => {
    if (!context.auth?.session) {
      throw new Error('Unauthorized: Authentication required');
    }

    // Assuming roles are stored in session or user object
    const userRole = context.auth.session.user?.role;
    if (userRole !== 'admin') {
      throw new Error('Forbidden: Admin access required');
    }

    return next({
      context: {
        ...context,
        auth: context.auth,
        userId: context.auth.session.userId,
        userRole: 'admin',
      },
    });
  });
}

export const adminRouter = os
  .use(ensureAdmin())
  .router({
    // User management - admin-only procedures
    listUsers: os
      .input(
        z.object({
          limit: z.number().default(20),
          offset: z.number().default(0),
          role: z.enum(['user', 'admin']).optional(),
        })
      )
      .handler(async ({ input }) => {
        // Admin-only logic - auth/role check in middleware
        return {
          items: [],
          total: 0,
          limit: input.limit,
          offset: input.offset,
        };
      }),

    getUser: os
      .input(z.object({ userId: z.string() }))
      .handler(async ({ input }) => {
        // Admin-only logic - auth/role check in middleware
        return {
          id: input.userId,
          email: 'user@example.com',
          role: 'user',
        };
      }),

    updateUserRole: os
      .input(
        z.object({
          userId: z.string(),
          role: z.enum(['user', 'admin']),
        })
      )
      .handler(async ({ input }) => {
        // Admin-only logic - auth/role check in middleware
        return { success: true, userId: input.userId, role: input.role };
      }),

    // System administration
    getSystemStats: os.handler(async () => {
      // Admin-only logic - auth/role check in middleware
      return {
        totalUsers: 0,
        totalSessions: 0,
      };
    }),
  });

---

## Middleware for Route Protection

### Middleware (`middleware.ts`)

Middleware only checks for auth token presence in headers. Route protection logic is handled in layouts and the auth middleware is on the API level.

import { NextRequest, NextResponse } from 'next/server';

// Simple token check utility
function hasAuthToken(request: NextRequest): boolean {
  const authHeader = request.headers.get('authorization');
  const token = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : null;
  return !!token;
}

export async function middleware(request: NextRequest) {
  const path = request.nextUrl.pathname;
  const hasToken = hasAuthToken(request);

  // Only pass token presence to context via header
  // Actual auth validation happens in server procedures
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set('x-has-auth-token', hasToken ? 'true' : 'false');

  return NextResponse.next({
    request: {
      headers: requestHeaders,
    },
  });
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};

---

## Route Layout Templates

### Private Routes Layout (`app/(private)/layout.tsx`)

import { redirect } from 'next/navigation';
import { getAuth } from '@/server/auth';

export default async function PrivateLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const auth = await getAuth();

  if (!auth?.session) {
    redirect('/login');
  }

  return (
    <div className="min-h-screen flex">
      <nav className="w-64 bg-gray-900 text-white p-4">
        {/* Navigation */}
        <a href="/dashboard">Dashboard</a>
        <a href="/profile">Profile</a>
        <a href="/settings">Settings</a>
      </nav>
      <main className="flex-1">{children}</main>
    </div>
  );
}

### Admin Routes Layout (`app/(admin)/layout.tsx`)

import { redirect } from 'next/navigation';
import { getAuth } from '@/server/auth';

export default async function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const auth = await getAuth();

  if (!auth?.session) {
    redirect('/login');
  }

  if (auth.session.user?.role !== 'admin') {
    redirect('/dashboard');
  }

  return (
    <div className="min-h-screen flex">
      <nav className="w-64 bg-gray-950 text-white p-4">
        {/* Admin Navigation */}
        <a href="/admin">Dashboard</a>
        <a href="/admin/users">Users</a>
        <a href="/admin/settings">Settings</a>
        <a href="/admin/logs">Logs</a>
      </nav>
      <main className="flex-1">{children}</main>
    </div>
  );
}

---

## OpenAPI Setup

### 1. Generate OpenAPI Spec (`server/orpc/openapi.ts`)

import { OpenAPIGenerator } from '@orpc/openapi';
import { ZodToJsonSchemaConverter } from '@orpc/zod/zod4';
import { orpc } from './index';

export async function generateOpenAPISpec() {
  const generator = new OpenAPIGenerator({
    schemaConverters: [new ZodToJsonSchemaConverter()],
  });

  const spec = await generator.generate(orpc, {
    info: {
      title: 'My App API',
      version: '1.0.0',
      description: 'Type-safe API with public, private, and admin routers',
    },
    servers: [
      {
        url: process.env.NEXT_PUBLIC_API_URL || '/api/orpc',
        description: 'API Server',
      },
    ],
  });

  return spec;
}

### 2. OpenAPI Route Handler (`app/api/openapi/spec/route.ts`)

import { generateOpenAPISpec } from '@/server/orpc/openapi';

export async function GET() {
  const spec = await generateOpenAPISpec();
  return Response.json(spec, {
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'public, max-age=3600',
    },
  });
}

### 3. Swagger UI Endpoint (Optional) (`app/api/openapi/swagger/route.ts`)

export async function GET() {
  return new Response(
    `<!DOCTYPE html>
    <html>
      <head>
        <title>API Documentation</title>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@3/swagger-ui.css">
      </head>
      <body>
        <div id="root"></div>
        <script src="https://unpkg.com/swagger-ui-dist@3/swagger-ui-bundle.js"></script>
        <script>
          SwaggerUIBundle({
            url: '/api/openapi/spec',
            dom_id: '#root',
            presets: [SwaggerUIBundle.presets.apis],
            layout: 'BaseLayout'
          })
        </script>
      </body>
    </html>`,
    {
      headers: { 'Content-Type': 'text/html; charset=utf-8' },
    }
  );
}

---

## Usage Examples

### Client Component with oRPC + React Query

'use client';

import { useQuery, useMutation } from '@tanstack/react-query';
import { orpcClient } from '@/client/orpc';

export function UserProfile() {
  // Fetch user profile
  const { data: profile, isLoading } = useQuery(
    orpcClient.private.getProfile.queryOptions()
  );

  // Update profile mutation
  const { mutate: updateProfile } = useMutation(
    orpcClient.private.updateProfile.mutationOptions()
  );

  if (isLoading) return <div>Loading...</div>;

  return (
    <div>
      <h1>Welcome, {profile?.name}</h1>
      <button onClick={() => updateProfile({ name: 'New Name' })}>
        Update Profile
      </button>
    </div>
  );
}

### Server Component with SSR Prefetching

import { dehydrate } from '@tanstack/react-query';
import { getQueryClient } from '@/app/get-query-client';
import { getServerORPC } from '@/client/orpc.server';
import { HydrationBoundary } from '@tanstack/react-query';
import { UserProfile } from '@/components/user-profile';

export default async function DashboardPage() {
  const queryClient = getQueryClient();
  const orpc = await getServerORPC();

  // Prefetch on server
  await queryClient.prefetchQuery({
    queryKey: ['profile'],
    queryFn: () => orpc.private.getProfile(),
  });

  const dehydratedState = dehydrate(queryClient);

  return (
    <HydrationBoundary state={dehydratedState}>
      <UserProfile />
    </HydrationBoundary>
  );
}

### Admin Page Example

'use client';

import { useQuery } from '@tanstack/react-query';
import { orpcClient } from '@/client/orpc';

export function UserManagement() {
  const { data: users, isLoading } = useQuery(
    orpcClient.admin.listUsers.queryOptions({
      input: { limit: 20 },
    })
  );

  if (isLoading) return <div>Loading users...</div>;

  return (
    <div>
      <h1>User Management</h1>
      <table>
        <tbody>
          {users?.items.map((user) => (
            <tr key={user.id}>
              <td>{user.email}</td>
              <td>{user.role}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

---

## Environment Variables

Add to your `.env.local`:

NEXT_PUBLIC_API_URL=http://localhost:3000/api/orpc
# Better Auth configuration
AUTH_SECRET=your-secret-key
DATABASE_URL=your-database-url

---

## Key Takeaways

- **oRPC Server**: Define routers with middleware for auth/admin checks
- **React Query Integration**: Use `@orpc/react-query` for automatic query options
- **SSR Support**: Prefetch queries on the server and dehydrate state
- **Route Protection**: Use middleware and layout redirects
- **Better Auth**: Integrate with existing auth config in `@/server/auth`
- **Type Safety**: Full end-to-end type safety from server to client

---

## Additional Resources

- [oRPC Documentation](https://orpc.io)
- [React Query Documentation](https://tanstack.com/query/latest)
- [Next.js App Router](https://nextjs.org/docs/app)
- [Better Auth Docs](https://www.better-auth.com)

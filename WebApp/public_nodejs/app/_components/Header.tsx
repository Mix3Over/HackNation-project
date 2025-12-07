import { Button } from "@bluevoid-test/ui/button";
import { headers } from "next/headers";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { getSession } from "@/lib/auth/server";

async function Header() {
  const session = await getSession();

  return (
    <div className="fixed top-0 left-0 z-40 h-14 w-full">
      <div className="flex h-full w-full items-center justify-between gap-6 border-b border-solid px-6 shadow">
        <div className="flex items-center gap-4">
          <img alt="" className="h-12" src="logo.png" />
          <span className="font-extrabold font-mono text-2xl">Admin123</span>
        </div>
        {/*<div>
					{!session ? (
						<Button
							variant="ghost"
							className="rounded-full border-stone-800 bg-stone-800 hover:bg-stone-700 hover:text-stone-50"
							formAction={async () => {
								"use server";
								const res = await auth.api.signInSocial({
									body: {
										provider: "discord",
										callbackURL: "/",
									},
								});
								if (!res.url) {
									throw new Error("No URL returned from signInSocial");
								}
								redirect(res.url);
							}}
						>
							Sign in with Discord
						</Button>
					) : (
						<Button
							className="rounded-full border-stone-800 bg-stone-800 hover:bg-stone-700 hover:text-stone-50"
							formAction={async () => {
								"use server";
								await auth.api.signOut({
									headers: await headers(),
								});
								redirect("/");
							}}
						>
							Sign out
						</Button>
					)}
				</div>*/}
      </div>
    </div>
  );
}

export default Header;

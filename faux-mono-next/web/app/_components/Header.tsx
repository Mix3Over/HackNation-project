import { Button } from "@bluevoid-test/ui/button";
import { headers } from "next/headers";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { getSession } from "@/lib/auth/server";

async function Header() {
	const session = await getSession();

	return (
		<div className="fixed top-0 left-0 z-40 h-14 w-full">
			<div className="flex h-full w-full items-center justify-between border-solid border-stone-800 border-b bg-stone-900 px-6 gap-6 text-stone-50">
				<div className="flex gap-4 items-center">
					<img src="logo.png" alt="" className="h-12" />
					<span className="font-mono font-extrabold text-2xl">Admin123</span>
				</div>
				<div>
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
				</div>
			</div>
		</div>
	);
}

export default Header;

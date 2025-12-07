"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

function Redir() {
	const router = useRouter();
	useEffect(() => {
		router.push("/");
	}, []);
	return null;
}

export default Redir;
